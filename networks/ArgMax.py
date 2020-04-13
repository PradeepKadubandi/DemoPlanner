import torch

class ArgMaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input, 1, keepdims=True)
        ctx._input_shape = input.shape
        ctx._input_dtype = input.dtype
        ctx._input_device = input.device
        ctx.save_for_backward(idx)
        return idx.float()

    @staticmethod
    def backward(ctx, grad_output):
        # print ('Size of input to argmax function backward: {}'.format(grad_output.size()))
        # print ('input to argmax function backward:')
        # print (grad_output)
        idx, = ctx.saved_tensors
        # print ('Saved index tensor:')
        # print (idx)
        grad_input = torch.zeros(ctx._input_shape, device=ctx._input_device, dtype=ctx._input_dtype)
        grad_input.scatter_(1, idx, grad_output)
        # print ('Output gradient from argmax function:')
        # print (grad_input)
        return grad_input