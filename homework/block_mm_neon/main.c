#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

void BlockMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result)
{
    //@@ Insert code to implement block matrix multiply here
    int blockSize = 4;
    for (int row = 0; row < result->shape[0]; row+=blockSize)
    {
        for (int col = 0; col < result->shape[1]; col+=blockSize)
        {
            int numBlockRowIters = ((row + blockSize) <= result->shape[0]) ? (row + blockSize) : (result->shape[0]);
            for (int blockRow = row; blockRow < numBlockRowIters; blockRow++)
            {
                int numBlockColIters = ((col + blockSize) <= result->shape[1]) ? (col + blockSize) : (result->shape[1]);
                for (int blockCol = col; blockCol < numBlockColIters; blockCol++)
                {
                    int numIters = ((input0->shape[1] / 4) * 4);
                    for (int i = 0; i < numIters; i+=4)
                    {
                        float data0Arr[4] = {0, 0, 0, 0};
                        data0Arr[0] = input0->data[blockRow * input0->shape[1] + i];
                        data0Arr[1] = input0->data[blockRow * input0->shape[1] + (i + 1)];
                        data0Arr[2] = input0->data[blockRow * input0->shape[1] + (i + 2)];
                        data0Arr[3] = input0->data[blockRow * input0->shape[1] + (i + 3)];

                        float32x4_t data0 = vld1q_f32(data0Arr);
                        
                        float data1Arr[4] = {0, 0, 0, 0};
                        data1Arr[0] = input1->data[i       * input1->shape[1] + blockCol];
                        data1Arr[1] = input1->data[(i + 1) * input1->shape[1] + blockCol];
                        data1Arr[2] = input1->data[(i + 2) * input1->shape[1] + blockCol];
                        data1Arr[3] = input1->data[(i + 3) * input1->shape[1] + blockCol];
                        
                        float32x4_t data1 = vld1q_f32(data1Arr);
                        
                        float32x4_t data = vmulq_f32(data0, data1);
                        result->data[blockRow * result->shape[1] + blockCol] += vaddvq_f32(data);
                    }
                    if ((input0->shape[1] - numIters) == 3)
                    {
                        float data0Arr[4] = {0, 0, 0, 0};
                        data0Arr[0] = input0->data[blockRow * input0->shape[1] + numIters];
                        data0Arr[1] = input0->data[blockRow * input0->shape[1] + (numIters + 1)];
                        data0Arr[2] = input0->data[blockRow * input0->shape[1] + (numIters + 2)];

                        float32x4_t data0 = vld1q_f32(data0Arr);
                        
                        float data1Arr[4] = {0, 0, 0, 0};
                        data1Arr[0] = input1->data[numIters       * input1->shape[1] + blockCol];
                        data1Arr[1] = input1->data[(numIters + 1) * input1->shape[1] + blockCol];
                        data1Arr[2] = input1->data[(numIters + 2) * input1->shape[1] + blockCol];
                        
                        float32x4_t data1 = vld1q_f32(data1Arr);
                        
                        float32x4_t data = vmulq_f32(data0, data1);
                        result->data[blockRow * result->shape[1] + blockCol] += vaddvq_f32(data);
                    }
                    else if ((input0->shape[1] - numIters) == 2)
                    {
                        float data0Arr[4] = {0, 0, 0, 0};
                        data0Arr[0] = input0->data[blockRow * input0->shape[1] + numIters];
                        data0Arr[1] = input0->data[blockRow * input0->shape[1] + (numIters + 1)];

                        float32x4_t data0 = vld1q_f32(data0Arr);
                        
                        float data1Arr[4] = {0, 0, 0, 0};
                        data1Arr[0] = input1->data[numIters       * input1->shape[1] + blockCol];
                        data1Arr[1] = input1->data[(numIters + 1) * input1->shape[1] + blockCol];
                        
                        float32x4_t data1 = vld1q_f32(data1Arr);
                        
                        float32x4_t data = vmulq_f32(data0, data1);
                        result->data[blockRow * result->shape[1] + blockCol] += vaddvq_f32(data);
                    }
                    else if ((input0->shape[1] - numIters) == 1)
                    {
                        result->data[blockRow * result->shape[1] + blockCol] += (input0->data[blockRow * input0->shape[1] + numIters] * input1->data[numIters * input1->shape[1] + blockCol]);
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];

    // Host input and output vectors and sizes
    Matrix host_a, host_b, host_c, answer;
    
    cl_int err;

    err = LoadMatrix(input_file_a, &host_a);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_c, &answer);
    CHECK_ERR(err, "LoadMatrix");

    int rows, cols;
    //@@ Update these values for the output rows and cols of the output
    //@@ Do not use the results from the answer matrix
    rows = host_a.shape[0];
    cols = host_b.shape[1];

    // Allocate the memory for the target.
    host_c.shape[0] = rows;
    host_c.shape[1] = cols;
    host_c.data = (float *)malloc(sizeof(float) * host_c.shape[0] * host_c.shape[1]);

    // Call your matrix multiply.
    BlockMatrixMultiply(&host_a, &host_b, &host_c);

    // Call to print the matrix
    //PrintMatrix(&host_c);

    // Check the result of the matrix multiply
    CheckMatrix(&answer, &host_c);

    // Save the matrix
    SaveMatrix(input_file_d, &host_c);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}