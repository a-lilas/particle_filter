#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <math.h>
#include <random>

class Particle{
    // 宣言(実装以外の部分)はhppファイルに書くことが多い
    public:
        int m_position_x;
        int m_position_y;
        int m_speed_x;
        int m_speed_y;
        double m_weight;
        double m_likelyhood;

        Particle(int);
};

class ParticleFilter{
    // 宣言(実装以外の部分)はhppファイルに書くことが多い
    public:
        int m_cnt;
        double m_ave_x;
        double m_ave_y;
        std::vector<Particle> m_particles;

        void Initialize();
        void Draw(cv::Mat);
        void Predict();
        void LikelyHood(cv::Mat);
        void Resampling();
        ParticleFilter(int);
};
