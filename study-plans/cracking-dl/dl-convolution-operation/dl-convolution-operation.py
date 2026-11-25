import numpy as np

def conv2d(input: list, filters: list, bias=None, padding: int = 0, stride: int = 1) -> list:
    values=np.asarray(input,dtype=np.float64); kernels=np.asarray(filters,dtype=np.float64)
    out_channels,in_channels,kernel_h,kernel_w=kernels.shape; _,height,width=values.shape
    padded=np.pad(values,((0,0),(padding,padding),(padding,padding)))
    out_h=(height+2*padding-kernel_h)//stride+1; out_w=(width+2*padding-kernel_w)//stride+1
    output=np.empty((out_channels,out_h,out_w),dtype=np.float64); offsets=np.zeros(out_channels) if bias is None else np.asarray(bias,dtype=np.float64)
    for channel in range(out_channels):
        for row in range(out_h):
            for column in range(out_w):
                patch=padded[:,row*stride:row*stride+kernel_h,column*stride:column*stride+kernel_w]
                output[channel,row,column]=np.sum(patch*kernels[channel])+offsets[channel]
    return np.round(output,4).tolist()