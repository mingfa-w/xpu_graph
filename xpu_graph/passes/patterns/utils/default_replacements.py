import torch


class DefaultSliceSumCatModule(torch.nn.Module):
    def __init__(self, slice_param):
        """
        Args:
            slice_param (list of tuples): A list of slice indices, where each tuple
                                          contains (start_idx, end_idx) for slicing.
        """
        super().__init__()

        slice_ = []
        for param in slice_param:
            slice_ += [param[0], param[1]]

        self.slice_param_list = slice_param

    def forward(self, input):
        """
        Forward pass for the SliceSumCatOperation.

        Args:
            input (torch.Tensor): The input tensor of shape (batch, row, col).

        Returns:
            torch.Tensor: The output tensor of shape (batch, len(slice_param) * col). The processed tensor after slice -> sum -> cat operations.
        """
        target_tensors = []
        for slice_arg in self.slice_param_list:
            slice_tensor = input[:, slice_arg[0] : slice_arg[1], :]
            sum_tensor = torch.sum(slice_tensor, dim=[1])
            target_tensors.append(sum_tensor)
        return torch.cat(target_tensors, axis=-1)
