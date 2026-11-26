import numpy as np

def transposed_conv2d(input: list, filters: list, bias=None, stride: int = 1, padding: int = 0) -> list:
    values=np.asarray(input,dtype=np.float64); kernels=np.asarray(filters,dtype=np.float64)
    in_channels,height,width=values.shape; _,out_channels,kernel_h,kernel_w=kernels.shape
    full_h=(height-1)*stride+kernel_h; full_w=(width-1)*stride+kernel_w
    full=np.zeros((out_channels,full_h,full_w),dtype=np.float64)
    for in_channel in range(in_channels):
        for row in range(height):
            for column in range(width):
                full[:,row*stride:row*stride+kernel_h,column*stride:column*stride+kernel_w]+=values[in_channel,row,column]*kernels[in_channel]
    output=full[:,padding:full_h-padding if padding else full_h,padding:full_w-padding if padding else full_w]
    if bias is not None: output+=np.asarray(bias,dtype=np.float64)[:,None,None]
    return np.round(output,4).tolist()