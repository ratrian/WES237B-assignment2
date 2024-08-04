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
                    int numIters = (input0->shape[1] / 4) * 4;
                    for (int i = 0; i < numIters; i+=4)
                    {
                        float32x4_t data0 = vld1q_f32(input0->data + blockRow * input0->shape[1] + i);
                        float32x4_t data1 = vld1q_f32(input1->data + i * input1->shape[1] + blockCol);
                        float32x4_t data = vmulq_f32(data0, data1);
                        result->data[blockRow * result->shape[1] + blockCol] += vaddvq_f32(data);
                    }
                    for (int i = numIters; i < input0->shape[0]; i++)
                    {
                        result->data[blockRow * result->shape[1] + blockCol] += input0->data[blockRow * input0->shape[1] + i] * input1->data[i * input1->shape[1] + blockCol];
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