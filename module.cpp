#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) 
{   // b: batch, h: head, N: sequence length, d: embeding deimnesion
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val)
{   // b: batch, h: head, N: sequence length, d: embeding deimnesion
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val;
}


// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {

            // Step 1: QK^T (store into QK_t)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float dot = 0.0f;
                    for (int k = 0; k < d; k++) {
                        float q = fourDimRead(Q, b, h, i, k, H, N, d);
                        float k_val = fourDimRead(K, b, h, j, k, H, N, d);
                        dot += q * k_val;
                    }
                    twoDimWrite(QK_t, i, j, N, dot);
                }
            }

            // Step 2: Apply softmax on QK_t
            for (int i = 0; i < N; i++) {
                float row_sum = 0.0f;

                // Compute exp
                for (int j = 0; j < N; j++) {
                    float val = exp(twoDimRead(QK_t, i, j, N));
                    twoDimWrite(QK_t, i, j, N, val);
                    row_sum += val;
                }

                // Normalize
                for (int j = 0; j < N; j++) {
                    float val = twoDimRead(QK_t, i, j, N) / row_sum;
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }

            // Step 3: Compute O = P * V
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < d; k++) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++) {
                        float p = twoDimRead(QK_t, i, j, N);
                        float v = fourDimRead(V, b, h, j, k, H, N, d);
                        sum += p * v;
                    }
                    fourDimWrite(O, b, h, i, k, H, N, d, sum);
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    int tileSize = 32;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {

            // Step 1: Blocked Matrix Multiply for QK_T
            for (int i0 = 0; i0 < N; i0 += tileSize) 
            {
                int i_max = std::min(i0 + tileSize, N);
                for (int j0 = 0; j0 < N; j0 += tileSize) 
                {
                    int j_max = std::min(j0 + tileSize, N);
                    for (int k0 = 0; k0 < d; k0 += tileSize)
                     {
                        int k_max = std::min(k0 + tileSize, d);

                        for (int i = i0; i < i_max; i++) 
                        {
                            for (int j = j0; j < j_max; j++) 
                            {
                                float sum = twoDimRead(QK_t, i, j, N); // accumulate
                                for (int k = k0; k < k_max; k++)
                                {
                                    float q = fourDimRead(Q, b, h, i, k, H, N, d);
                                    float k_val = fourDimRead(K, b, h, j, k, H, N, d);
                                    sum += q * k_val;
                                }
                                twoDimWrite(QK_t, i, j, N, sum);
                            }
                        }
                    }
                }
            }

            // Step 2: Softmax over rows
            for (int i = 0; i < N; i++) {
                float row_sum = 0.0f;
                for (int j = 0; j < N; j++) 
                {
                    float val = exp(twoDimRead(QK_t, i, j, N));
                    twoDimWrite(QK_t, i, j, N, val);
                    row_sum += val;
                }
                for (int j = 0; j < N; j++) 
                {
                    float val = twoDimRead(QK_t, i, j, N) / row_sum;
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }

            // Step 3: Blocked Matix Multiply for O = P * V
            for (int i0 = 0; i0 < N; i0 += tileSize) 
            {
                int i_max = std::min(i0 + tileSize, N);
                for (int k0 = 0; k0 < d; k0 += tileSize)
                {
                    int k_max = std::min(k0 + tileSize, d);
                    for (int j0 = 0; j0 < N; j0 += tileSize) 
                    {
                        int j_max = std::min(j0 + tileSize, N);

                        for (int i = i0; i < i_max; i++) 
                        {
                            for (int k = k0; k < k_max; k++) 
                            {
                                float sum = fourDimRead(O, b, h, i, k, H, N, d);
                                for (int j = j0; j < j_max; j++) 
                                {
                                    float p = twoDimRead(QK_t, i, j, N);
                                    float v = fourDimRead(V, b, h, j, k, H, N, d);
                                    sum += p * v;
                                }
                                fourDimWrite(O, b, h, i, k, H, N, d, sum);
                            }
                        }
                    }
                }
            }
        }
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < N; i++) {

                std::vector<float> scores(N);
                float max_score = -INFINITY;
                float sum = 0.0f;

                // Step 1: Compute QK_T for row i
                for (int j = 0; j < N; j++) {
                    float dot = 0.0f;
                    for (int k = 0; k < d; k++) {
                        dot += fourDimRead(Q, b, h, i, k, H, N, d) * fourDimRead(K, b, h, j, k, H, N, d);
                    }
                    scores[j] = dot;
                    if (dot > max_score) max_score = dot;
                }

                // Step 2: Softmax
                for (int j = 0; j < N; j++) {
                    scores[j] = exp(scores[j] - max_score);
                    sum += scores[j];
                }
                for (int j = 0; j < N; j++) {
                    scores[j] /= sum;
                }

                // Step 3: Compute P × V
                for (int k = 0; k < d; k++) {
                    float out = 0.0f;
                    for (int j = 0; j < N; j++) {
                        out += scores[j] * fourDimRead(V, b, h, j, k, H, N, d);
                    }
                    fourDimWrite(O, b, h, i, k, H, N, d, out);
                }
            }
        }
    }
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {

            for (int jBlock = 0; jBlock < N; jBlock += Bc) {
                for (int iBlock = 0; iBlock < N; iBlock += Br) {

                    // Compute Pij (exp(Q × K^T))
                    for (int j = 0; j < std::min(Bc, N - jBlock); j++) 
                    {
                        int colIdx = jBlock + j;
                        for (int i = 0; i < std::min(Br, N - iBlock); i++) 
                        {
                            int rowIdx = iBlock + i;

                            float dot = 0.0f;
                            for (int k = 0; k < d; k++) 
                            {
                                float q = fourDimRead(Q, b, h, rowIdx, k, H, N, d);
                                float k_val = fourDimRead(K, b, h, colIdx, k, H, N, d);
                                dot += q * k_val;
                            }

                            float expDot = exp(dot);
                            twoDimWrite(Pij, i, j, Bc, expDot);
                        }
                    }

                    // Compute softmax(Pij), then accumulate Pij * V
                    for (int i = 0; i < std::min(Br, N - iBlock); i++) 
                    {
                        int actualRow = i + iBlock;
                        float sumP = 0.0f;

                        for (int j = 0; j < std::min(Bc, N - jBlock); j++) 
                        {
                            sumP += twoDimRead(Pij, i, j, Bc);
                        }

                        float l_old = l[actualRow];
                        float l_new = l_old + sumP;

                        for (int k = 0; k < d; k++) 
                        {
                            float acc = 0.0f;

                            for (int j = 0; j < std::min(Bc, N - jBlock); j++) 
                            {
                                int colIdx = j + jBlock;
                                float pij = twoDimRead(Pij, i, j, Bc);
                                float v = fourDimRead(V, b, h, colIdx, k, H, N, d);
                                acc += pij * v;
                            }

                            float o_prev = fourDimRead(O, b, h, actualRow, k, H, N, d);
                            float o_next = (o_prev * l_old + acc) / l_new;
                            fourDimWrite(O, b, h, actualRow, k, H, N, d, o_next);
                        }

                        l[actualRow] = l_new;
                    }
                }
            }

            std::fill(l.begin(), l.end(), 0.0f); // Reset normalization vector
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
