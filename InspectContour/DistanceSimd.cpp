#include "pch.h"
#include "DistanceSimd.h"


float L1DistSSE2(const float& xt, const float& yt, const float* x_i, const float* y_i, const int& length, float min_distance) {
    __m128 x0 = _mm_set1_ps(xt);
    __m128 y0 = _mm_set1_ps(yt);

    int padded_length = (length + 3) & ~3;
    float* padded_x = new float[padded_length];
    float* padded_y = new float[padded_length];

    std::memcpy(padded_x, x_i, length * sizeof(float));
    std::memcpy(padded_y, y_i, length * sizeof(float));

    //std::memset(padded_x + length, x_i[0], (padded_length - length) * sizeof(float));
    //std::memset(padded_y + length, y_i[0], (padded_length - length) * sizeof(float));

    std::fill(padded_x + length, padded_x + padded_length, x_i[0]);
    std::fill(padded_y + length, padded_y + padded_length, y_i[0]);

    for (int i = 0; i < padded_length; i += 4) {
        __m128 x_vec4 = _mm_loadu_ps(padded_x + i);
        __m128 y_vec4 = _mm_loadu_ps(padded_y + i);

        x_vec4 = _mm_sub_ps(x_vec4, x0);
        y_vec4 = _mm_sub_ps(y_vec4, y0);

        x_vec4 = _mm_and_ps(x_vec4, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
        y_vec4 = _mm_and_ps(y_vec4, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));

        __m128 d = _mm_add_ps(x_vec4, y_vec4);

        __m128 min = _mm_min_ps(d, _mm_shuffle_ps(d, d, _MM_SHUFFLE(0, 1, 2, 3)));
        min = _mm_min_ps(min, _mm_shuffle_ps(min, min, _MM_SHUFFLE(1, 0, 3, 2)));

        min_distance = min(_mm_cvtss_f32(min), min_distance);
        /*
        float d4[4];
        _mm_storeu_ps(d4, d);
        for (int i = 0; i < 4; ++i) {
            min_distance = std::min(min_distance, d4[i]);
        }*/
    }

    return min_distance;
}


float L2DistSSE2(const float& xt, const float& yt, const float* x_i, const float* y_i, const int& length, float min_distance) {
    __m128 x0 = _mm_set1_ps(xt);
    __m128 y0 = _mm_set1_ps(yt);

    int padded_length = (length + 3) & ~3;
    float* padded_x = new float[padded_length];
    float* padded_y = new float[padded_length];

    std::memcpy(padded_x, x_i, length * sizeof(float));
    std::memcpy(padded_y, y_i, length * sizeof(float));

    std::fill(padded_x + length, padded_x + padded_length, x_i[0]);
    std::fill(padded_y + length, padded_y + padded_length, y_i[0]);


    for (int i = 0; i < padded_length; i += 4) {
        __m128 x_vec4 = _mm_loadu_ps(padded_x + i);
        __m128 y_vec4 = _mm_loadu_ps(padded_y + i);

        x_vec4 = _mm_sub_ps(x_vec4, x0);
        y_vec4 = _mm_sub_ps(y_vec4, y0);

        x_vec4 = _mm_mul_ps(x_vec4, x_vec4);
        y_vec4 = _mm_mul_ps(y_vec4, y_vec4);

        __m128 d_square = _mm_add_ps(x_vec4, y_vec4);

        __m128 min = _mm_min_ps(d_square, _mm_shuffle_ps(d_square, d_square, _MM_SHUFFLE(0, 1, 2, 3)));
        min = _mm_min_ps(min, _mm_shuffle_ps(min, min, _MM_SHUFFLE(1, 0, 3, 2)));

        min_distance = min(min_distance, (float)_mm_cvtss_f32(min));
    }
    return min_distance;
}


float L2DistAVX(const float& xt, const float& yt, const float* x_i, const float* y_i, const int& length, float min_distance) {
    __m256 x0 = _mm256_set1_ps(xt);
    __m256 y0 = _mm256_set1_ps(yt);

    int padded_length = (length + 7) & ~7;
    float* padded_x = new float[padded_length];
    float* padded_y = new float[padded_length];

    std::memcpy(padded_x, x_i, length * sizeof(float));
    std::memcpy(padded_y, y_i, length * sizeof(float));

    std::fill(padded_x + length, padded_x + padded_length, x_i[0]);
    std::fill(padded_y + length, padded_y + padded_length, y_i[0]);

    for (int i = 0; i < padded_length; i += 8) {
        __m256 x_vec8 = _mm256_loadu_ps(padded_x + i);
        __m256 y_vec8 = _mm256_loadu_ps(padded_y + i);

        x_vec8 = _mm256_sub_ps(x_vec8, x0);
        y_vec8 = _mm256_sub_ps(y_vec8, y0);

        x_vec8 = _mm256_mul_ps(x_vec8, x_vec8);
        y_vec8 = _mm256_mul_ps(y_vec8, y_vec8);

        __m256 d_square = _mm256_add_ps(x_vec8, y_vec8);

        __m256 min = _mm256_min_ps(d_square, _mm256_permute2f128_ps(d_square, d_square, 1));
        min = _mm256_min_ps(min, _mm256_shuffle_ps(min, min, _MM_SHUFFLE(2, 3, 0, 1)));
        min = _mm256_min_ps(min, _mm256_shuffle_ps(min, min, _MM_SHUFFLE(1, 0, 3, 2)));
        __m128 reg128 = _mm256_extractf128_ps(min, 0);
        min_distance = min(min_distance, _mm_cvtss_f32(reg128));
    }
    return min_distance;
}