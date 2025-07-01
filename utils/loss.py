import torch
import torch.nn as nn
import pytorch_msssim

class HuberLoss(nn.Module):
    """
    Huber Loss, a robust loss function that is less sensitive to outliers than MSE.
    It's L1 for large errors and L2 for small errors.
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss = torch.abs(pred - target)
        # The mask identifies small errors (where L2 loss is used)
        mask = l1_loss < self.delta
        
        # L2 loss for small errors
        sq_loss = 0.5 * (l1_loss ** 2)
        # L1-like loss for large errors
        abs_loss = self.delta * (l1_loss - 0.5 * self.delta)

        return torch.mean(torch.where(mask, sq_loss, abs_loss))

class Loss(nn.modules.loss._Loss):
    """
    A wrapper class for combining multiple loss functions with specified weights.
    Parses a loss string (e.g., '1.0*L1+0.1*SSIM') to build the final loss.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss = []
        self.loss_module = nn.ModuleList()

        # --- Parse the loss string ---
        for loss_term in args.loss.split('+'):
            weight, loss_type = loss_term.split('*')
            
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'Huber':
                # You can adjust the delta value here if needed
                loss_function = HuberLoss(delta=0.5)
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'SSIM':
                # data_range should match the expected range of the input tensors
                loss_function = pytorch_msssim.SSIM(data_range=1.0, size_average=True, channel=1)
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })

        # Add a placeholder for the total loss for logging purposes
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 1, 'function': None})

        # Register all non-None loss functions as submodules
        for term in self.loss:
            if term['function'] is not None:
                print(f"Preparing loss: {term['weight']:.3f} * {term['type']}")
                self.loss_module.append(term['function'])

        device = torch.device('cuda' if args.cuda else 'cpu')
        self.loss_module.to(device)

        if args.cuda and torch.cuda.device_count() > 1:
            self.loss_module = nn.DataParallel(self.loss_module)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculates the total weighted loss.

        Args:
            pred (torch.Tensor): The predicted tensor, shape (B, C, D, H, W).
            target (torch.Tensor): The ground truth tensor, shape (B, C, D, H, W).
        
        Returns:
            A tuple containing:
            - total_loss (torch.Tensor): The final weighted loss.
            - losses (dict): A dictionary of individual weighted loss components.
        """
        total_loss = 0
        losses = {}

        for i, term in enumerate(self.loss):
            if term['function'] is not None:
                # Get the actual loss function from the module list
                loss_func = self.loss_module.module[i] if isinstance(self.loss_module, nn.DataParallel) else self.loss_module[i]

                if term['type'] == 'SSIM':
                    # SSIM is designed for 2D images. We treat the depth dimension as part of the batch.
                    # Reshape from (B, C, D, H, W) to (B*D, C, H, W)
                    b, c, d, h, w = pred.shape
                    pred_reshaped = pred.view(b * d, c, h, w)
                    target_reshaped = target.view(b * d, c, h, w)
                    
                    # SSIM value is similarity (higher is better), so loss is 1 - SSIM
                    _loss = 1.0 - loss_func(pred_reshaped, target_reshaped)
                else:
                    # L1, MSE, Huber work on any tensor shape directly
                    _loss = loss_func(pred, target)
                
                effective_loss = term['weight'] * _loss
                losses[term['type']] = effective_loss
                total_loss += effective_loss
        
        if len(losses) > 1:
            losses['Total'] = total_loss

        return total_loss, losses