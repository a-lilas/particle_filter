#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>
#include <random>
#include "particle_filter.hpp"

#define MAX_WIDTH 1280
#define MAX_HEIGHT 720
#define MAX_SPEED 30
#define MIN_SPEED -30

// TODO: segmentation faultが発生する場合があるため改善する必要
// TODO: 早めの移動は苦手なため改善する必要
// TODO: 照明変化などに頑健にするためHSV色空間に対応させる必要

Particle::Particle(int particle_cnt){
    m_weight = 1.0 / particle_cnt;
    m_speed_x = 0;
    m_speed_y = 0;
}

ParticleFilter::ParticleFilter(int particle_cnt){
    // コンストラクタ
    // 粒子数
    m_cnt = particle_cnt;
    // 粒子を表す変数の宣言
    m_particles.reserve(m_cnt);
    for(int i=0; i<m_cnt; i++){
        m_particles.push_back(Particle(m_cnt));
    }
}

void ParticleFilter::Initialize(){
    // 粒子の初期化(ランダムに粒子を画像中に配置)
    int i;
    // 非決定的な乱数生成器を生成
    std::random_device rnd;
    //  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<> rand_width(0, MAX_WIDTH);
    std::uniform_int_distribution<> rand_height(0, MAX_HEIGHT);

    for(i=0; i<m_cnt; i++){
        m_particles[i].m_position_y = rand_width(mt);
        m_particles[i].m_position_x = rand_height(mt);
    }
}

void ParticleFilter::Draw(cv::Mat frame){
    // 粒子の描画処理
    // 描画する粒子数
    int draw_cnt = 500;
    for(int i=0; i<draw_cnt; i++){
        cv::circle(frame, cv::Point(m_particles[i].m_position_y, m_particles[i].m_position_x), 1, cv::Scalar(0, 200, 0), -1, CV_AA);
    }
    cv::circle(frame, cv::Point(m_ave_y, m_ave_x), 8, cv::Scalar(0, 0, 200), -1, CV_AA);
}

void ParticleFilter::Predict(){
    // 粒子の予測(システム)モデル
    // 等速直線運動にガウスノイズを加算したものを仮定
    std::random_device rnd;
    std::mt19937 mt(rnd());
    // 一様分布からノイズを発生させる
    std::uniform_int_distribution<> noise(5.0, 5.0);
    std::uniform_int_distribution<> speed_noise(-20.0, 20.0);
    for(int i=0; i<m_cnt; i++){
        // キャストを用いて四捨五入
        // TODO: ガウスノイズが偏っている？
        // 和と差をそれぞれ用いて偏りを抑えている / ノイズの範囲に対して過敏
        m_particles[i].m_position_x += (int)(m_particles[i].m_speed_x + noise(mt) + 0.5);
        m_particles[i].m_position_y += (int)(m_particles[i].m_speed_y + noise(mt) + 0.5);
        m_particles[i].m_speed_x -= speed_noise(mt);
        m_particles[i].m_speed_y -= speed_noise(mt);

        if(m_particles[i].m_position_x > MAX_HEIGHT) m_particles[i].m_position_x = MAX_HEIGHT;
        if(m_particles[i].m_position_y > MAX_WIDTH) m_particles[i].m_position_y = MAX_WIDTH;
        if(m_particles[i].m_position_x < 0) m_particles[i].m_position_x = 0;
        if(m_particles[i].m_position_y < 0) m_particles[i].m_position_y = 0;
        if(m_particles[i].m_speed_x > MAX_SPEED) m_particles[i].m_speed_x = MAX_SPEED;
        if(m_particles[i].m_speed_y > MAX_SPEED) m_particles[i].m_speed_y = MAX_SPEED;
        if(m_particles[i].m_speed_x < MIN_SPEED) m_particles[i].m_speed_x = MIN_SPEED;
        if(m_particles[i].m_speed_y < MIN_SPEED) m_particles[i].m_speed_y = MIN_SPEED;

    }
}

void ParticleFilter::LikelyHood(cv::Mat frame){
    // 尤度の計算,粒子の重み付け,重み付き平均の算出
    cv::Vec3b intensity;
    double tmp_likelyhood;
    double tmp_sum = 0.0;

    for(int i=0; i<m_cnt; i++){
        // 輝度の取得
        intensity = frame.at<cv::Vec3b>(m_particles[i].m_position_x, m_particles[i].m_position_y);
        // 青色物体の追跡
        tmp_likelyhood = 1.0 / sqrt(pow((255 - intensity[0]), 2) + pow((0 - intensity[1]), 2) + pow((0 - intensity[2]), 2));
        // TODO: RGB色空間ではなくHSV色空間で処理(HSV=[230, 200, xxx]とした)
        // tmp_likelyhood = 1.0 / sqrt(pow((230 - intensity[0]), 2) + pow((200 - intensity[1]), 2));
        if(tmp_likelyhood == INFINITY){
            tmp_likelyhood = 0.000001;
        }
        m_particles[i].m_weight = tmp_likelyhood;
        tmp_sum += tmp_likelyhood;
    }

    m_ave_x = 0.0;
    m_ave_y = 0.0;

    for(int i=0; i<m_cnt; i++){
        m_particles[i].m_weight /= tmp_sum;
        // 重み付き平均の算出
        m_ave_x += m_particles[i].m_position_x * m_particles[i].m_weight;
        m_ave_y += m_particles[i].m_position_y * m_particles[i].m_weight;
    }
}

void ParticleFilter::Resampling(){
    // 粒子のリサンプリング
    // 参考: https://robotics.stackexchange.com/questions/479/particle-filters-how-to-do-resamplingss
    // 重みの累積和を計算した後，0~1でランダムに数を生成し，その範囲にあてはまるインデックスの粒子を発生させる．(ルーレット選択)

    std::vector<int> tmp_particles_position_x;
    std::vector<int> tmp_particles_position_y;
    std::vector<int> tmp_particles_speed_x;
    std::vector<int> tmp_particles_speed_y;

    tmp_particles_position_x.reserve(m_cnt);
    tmp_particles_position_y.reserve(m_cnt);
    tmp_particles_speed_x.reserve(m_cnt);
    tmp_particles_speed_y.reserve(m_cnt);

    for(int i=1; i<m_cnt; i++){
        // 累積和の計算
        m_particles[i].m_weight += m_particles[i-1].m_weight;
    }

    for(int i=0; i<m_cnt; i++){
        tmp_particles_position_x.push_back(m_particles[i].m_position_x);
        tmp_particles_position_y.push_back(m_particles[i].m_position_y);
        tmp_particles_speed_x.push_back(m_particles[i].m_speed_x);
        tmp_particles_speed_y.push_back(m_particles[i].m_speed_y);
    }

    // 一様乱数の生成
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> resampling_rnd(0.0,1.0);
    double resampling_index;
    
    // バイナリサーチで該当するインデックスを求める
    for(int i=0; i<m_cnt; i++){
        resampling_index = resampling_rnd(mt);
        int min_index = 0;
        int max_index = m_cnt - 1;
        while(1){
            int pivot = (min_index + max_index) / 2;
            if(m_particles[pivot].m_weight < resampling_index){
                min_index = pivot;
            }else{
                max_index = pivot;                
            }
            if(max_index - min_index <= 1) break;
        }
        m_particles[i].m_position_x = tmp_particles_position_x[max_index];
        m_particles[i].m_position_y = tmp_particles_position_y[max_index];
        m_particles[i].m_speed_x = tmp_particles_speed_x[max_index];
        m_particles[i].m_speed_y = tmp_particles_speed_y[max_index];
    }

}

int main(int argh, char* argv[]){
    //デバイスのオープン
    cv::VideoCapture cap(0);

    if(!cap.isOpened()){
        //カメラデバイスが正常にオープンしたか確認
        //読み込みに失敗したときの処理
        return -1;
    }

    // フレームサイズの指定
    cap.set(cv::CAP_PROP_FRAME_WIDTH, MAX_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, MAX_HEIGHT);

    cv::Mat frame;
    // 出力用
    cv::Mat output_frame;
    // cv::Mat hsvframe;

    // get a new frame from camera
    cap >> frame;


    ParticleFilter pf(5000);
    pf.Initialize();

    while(1){
        // get a new frame from camera
        cap >> frame;
        cap >> output_frame;
        // cv::cvtColor(frame, hsvframe, CV_RGB2HSV);
        
        pf.Draw(output_frame);
        pf.Predict();
        pf.LikelyHood(frame);
        pf.Resampling();

        //画像を表示(描画した粒子は画像中の画素として扱われる)
        cv::imshow("window", output_frame);

        int key = cv::waitKey(1);
        if(key == 113){
            //qボタンが押されたとき
            //whileループから抜ける
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}
