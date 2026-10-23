def conv2d(image, kernel, stride=1, padding=0):
    height, width = len(image), len(image[0])
    kernel_height, kernel_width = len(kernel), len(kernel[0])
    padded = [[0.0] * (width + 2 * padding) for _ in range(height + 2 * padding)]
    for row in range(height):
        for column in range(width):
            padded[row + padding][column + padding] = image[row][column]
    output_height = (len(padded) - kernel_height) // stride + 1
    output_width = (len(padded[0]) - kernel_width) // stride + 1
    output = []
    for row in range(output_height):
        output_row = []
        for column in range(output_width):
            value = 0.0
            for kernel_row in range(kernel_height):
                for kernel_column in range(kernel_width):
                    value += padded[row * stride + kernel_row][column * stride + kernel_column] * kernel[kernel_row][kernel_column]
            output_row.append(value)
        output.append(output_row)
    return output