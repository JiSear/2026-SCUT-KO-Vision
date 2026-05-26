#include <opencv2/opencv.hpp>
#include <windows.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#ifndef CAP_PROP_AUTOWB
#define CAP_PROP_AUTOWB 44
#endif

using namespace cv;
using namespace std;

HANDLE hSerial = INVALID_HANDLE_VALUE;

void initSerial(const char* port = "COM7") {
    hSerial = CreateFileA(port, GENERIC_WRITE, 0, 0, OPEN_EXISTING, 0, 0);
    if (hSerial == INVALID_HANDLE_VALUE) {
        cout << "[Warning] Serial open failed: " << port << endl;
    }
    else {
        DCB dcbSerialParams = { 0 };
        dcbSerialParams.DCBlength = sizeof(dcbSerialParams);

        if (GetCommState(hSerial, &dcbSerialParams)) {
            dcbSerialParams.BaudRate = CBR_115200;
            dcbSerialParams.ByteSize = 8;
            dcbSerialParams.StopBits = ONESTOPBIT;
            dcbSerialParams.Parity = NOPARITY;
            SetCommState(hSerial, &dcbSerialParams);
        }
        
        // 【防卡死护盾机制：串口非阻塞超时配置】
        // Windows的 WriteFile 默认是无限阻塞的。如果单片机死机或者下位机接收缓存溢出，
        // C++ 的画面会彻底定格。增加这几行能保证哪怕串口出问题，画面也能保持 30 帧常亮！
        COMMTIMEOUTS timeouts = { 0 };
        timeouts.WriteTotalTimeoutConstant = 10;   // 写入最大等待 10ms，超了强制跳过不卡线程
        timeouts.WriteTotalTimeoutMultiplier = 0;
        SetCommTimeouts(hSerial, &timeouts);
        
        cout << "[Success] Serial opened: " << port << endl;
    }
}

void sendAngles(float angle_x, float angle_y) {
    angle_x = max(min(angle_x, 45.0f), -45.0f);
    angle_y = max(min(angle_y, 45.0f), -45.0f);

    char buf[64];
    int x_val = (int)round(angle_x * 10);
    int y_val = (int)round(angle_y * 10);
    sprintf_s(buf, "#X%dY%d!", x_val, y_val);

    DWORD bytes = 0;
    if (hSerial != INVALID_HANDLE_VALUE)
        WriteFile(hSerial, buf, (DWORD)strlen(buf), &bytes, NULL);
    else
        cout << "SEND: " << buf << endl;
}

struct ColorDef {
    string name;
    Scalar lower;
    Scalar upper;
};

vector<ColorDef> colorList =
{
    
   {"Magenta", Scalar(128, 135, 42), Scalar(179, 255, 255)},
 //{"DarkPurple", Scalar(125, 40, 0), Scalar(179, 180, 255)},
   {"LightGreen", Scalar(33, 95, 45), Scalar(80, 255, 255)},
   {"Red", Scalar(150, 60, 100), Scalar(179, 255, 255)},
   {"LightPink", Scalar(128, 21, 60), Scalar(170, 255, 225)},
   {"GoldenYellow", Scalar(10, 160, 101), Scalar(30, 255, 255)},
 //{"LightOrange", Scalar(0, 80, 137), Scalar(26, 132, 255)},
   {"RoyalBlue", Scalar(85, 88, 61), Scalar(125, 255, 255)},
 //{"SkyBlue", Scalar(89, 56, 61), Scalar(158, 183, 189)},
   {"GrassGreen", Scalar(46, 85, 31), Scalar(95, 255, 255)},
   {"LemonYellow", Scalar(26, 50, 92), Scalar(47, 255, 255)},

};

int numColors = (int)colorList.size();

struct Sector {
    int colorIdx;
    Point center;
    double area;
};

Point detectCenter(const Mat& frame) {
    Mat gray, blur;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur, Size(9, 9), 2);
    vector<Vec3f> circles;
    HoughCircles(blur, circles, HOUGH_GRADIENT, 1, 60, 100, 30, 8, 80);

    if (circles.empty()) return { -1, -1 };

    Point fc(frame.cols / 2, frame.rows / 2);
    int best = 0;
    double min_d = 1e9;
    for (int i = 0; i < circles.size(); ++i) {
        Point c(cvRound(circles[i][0]), cvRound(circles[i][1]));
        double d = norm(c - fc);
        if (d < min_d) {
            min_d = d;
            best = i;
        }
    }
    return Point(cvRound(circles[best][0]), cvRound(circles[best][1]));
}

int detectCentralColor(const Mat& frame, Point center) {
    // 【核心识别：中心采样域提取】
    // 设置采样半径为14，截取28x28的中心方形ROI。
    // 获取足够纯色面积以过滤噪波。
    int r = 14; 
    Rect roi(center.x - r, center.y - r, 2 * r, 2 * r);
    roi &= Rect(0, 0, frame.cols, frame.rows);
    if (roi.width <= 0 || roi.height <= 0) return -1;

    Mat hsv;
    cvtColor(frame(roi), hsv, COLOR_BGR2HSV);
    vector<int> cnt(numColors, 0);

    for (int i = 0; i < numColors; ++i) {
        Mat mask;
        inRange(hsv, colorList[i].lower, colorList[i].upper, mask);
        cnt[i] = countNonZero(mask);
    }

    int best = max_element(cnt.begin(), cnt.end()) - cnt.begin();

    // 【曝光阈值防呆机制】
    // 强制验证采样区该颜色的有效像素点必须大于 40 （约占整个28x28框区域的小部分）。
    // 在极限低曝光(-6)时，这一阈值可以有效屏蔽背景随机孤立高亮噪点或小面积反光。
    return cnt[best] > 40 ? best : -1;
}

Point detectCenterRobust(const Mat& frame, Point last_center, int target_color) {
    if (target_color >= 0) {
        Mat hsv, mask;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, colorList[target_color].lower, colorList[target_color].upper, mask);

        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1, -1), 1);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1, -1), 1);

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        struct BlobInfo { double area; Point p; };
        vector<BlobInfo> blobs;

        for (auto& c : contours) {
            double area = contourArea(c);
            // 放宽到一个极大的安全阈值（例如 25000，防止误认背景暗区）
            if (area < 20 || area > 25000) continue;
            Moments m = moments(c);
            if (m.m00 < 1) continue;
            blobs.push_back({ area, Point(cvRound(m.m10 / m.m00), cvRound(m.m01 / m.m00)) });
        }

        sort(blobs.begin(), blobs.end(), [](const BlobInfo& a, const BlobInfo& b) {
            return a.area > b.area;
            });

        if (last_center.x >= 0) {
            int best_idx = -1;
            double min_d = 1e9;
            for (int i = 0; i < blobs.size(); ++i) {
                double d = norm(blobs[i].p - last_center);
                if (d < min_d) {
                    min_d = d;
                    best_idx = i;
                }
            }
            // 只要距离上一帧的中心不超过 40 像素，就可以接续跟踪 (有效防止圆心跳跃到背景方灯上)
            if (best_idx >= 0 && min_d < 40) return blobs[best_idx].p;
        }

        if (blobs.size() >= 2) {
            //中央的圆形大圆盘比四周贴的方形色块面积要大得多。
            // 因此，如果同色块有多个，排名第一大（blobs[0]）的才是真正的中心圆。
            return blobs[0].p;
        }
        // 注意：如果只剩一个符合条件的色块，不能返回,因为它极大概率是面积大、更易被识别的方形装甲板。
        // 直接 fall through 到最后，交给下面原生的 detectCenter(frame) 依靠形状去重新寻找圆形！
    }

    return detectCenter(frame);
}

bool detectTargetSector(const Mat& frame, Point center, int target_color, Sector& out_sector) {
    if (target_color < 0) return false;

    Mat hsv, mask;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    inRange(hsv, colorList[target_color].lower, colorList[target_color].upper, mask);

    // 【减小开运算核】使用3x3软过滤防止拖影拉丝导致目标断裂，5x5闭运算填补孔洞
    Mat kernel_open = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat kernel_close = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel_open, Point(-1, -1), 1);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel_close, Point(-1, -1), 1);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    struct BlobInfo { double area; Point p; };
    vector<BlobInfo> blobs;

    for (auto& c : contours) {
        double area = contourArea(c);
        // 收紧最大面积限制，防止把整个黑色背景当成目标
        if (area < 50 || area > 25000) continue;

        Moments m = moments(c);
        if (m.m00 < 1) continue;
        blobs.push_back({ area, Point(cvRound(m.m10 / m.m00), cvRound(m.m01 / m.m00)) });
    }

    // 【致命修复】同色块区分！统一用面积排序彻底制霸
    // 根本不需要什么宽高比、也不需要用黑笔去黑掉中心圆来防止干扰！
    // 因为最大的那个一定是打击目标靶标(方形)！
    sort(blobs.begin(), blobs.end(), [](const BlobInfo& a, const BlobInfo& b) {
        return a.area > b.area;
        });

    for (int i = 0; i < (int)blobs.size(); ++i) {
        // 【修改识别距离结界：严防背景】目标装甲板距离中心也不能太远，强行收紧至 150 像素！
        // 把转盘轨道外的所有发光点直接掐死在摇篮里！
        double dist = norm(blobs[i].p - center);
        if (dist > 40 && dist < 150) {
            out_sector = { target_color, blobs[i].p, blobs[i].area };
            return true;
        }
    }
    return false;
}

// ===================== [修正：计算相对相机画面中心的角度误差] =====================
// 对于眼在手（相机在云台上）的结构，需要将目标移动到画面中心
void calcAngles(Point target, float& ax, float& ay) {
    float fx = 600, fy = 600;
    // 画面中心 (320, 240)
    float dx = target.x - 320.0f;
    float dy = target.y - 240.0f;

    // 【修复】反转云台舵机追踪角度的正负号，解决反向逃逸（正反馈）的问题
    ax = -atan2(dx, fx) * 180.0f / CV_PI;
    ay = -atan2(dy, fy) * 180.0f / CV_PI;


    // 【关键优化】彻底移除人为的 "死区(Deadzone)"
    // 由于能量机关始终在旋转，加死区会导致云台在追上目标后突然停转（刹车），瞬间又被甩开，从而产生肉眼可见的“一卡一卡”的抽搐。
    // 去除后微小的误差波动将能由电控端融合为平滑的角速率控制。
}

Point last_target{ -1,-1 };
Point smooth(Point p) {
    if (last_target.x < 0 || norm(last_target - p) > 50) {
        last_target = p;
    }
    Point s;
    s.x = (last_target.x * 1 + p.x * 2) / 3;
    s.y = (last_target.y * 1 + p.y * 2) / 3;
    last_target = s;
    return s;
}

int main() {
    VideoCapture cap;

    // 【赛场热插拔机制：死等外部摄像头】
    // 去掉对笔记本自带摄像头(0)的后备调用，强行死等索引 1 (外部相机)
    cout << "[System] 正在等待插入外部相机线..." << endl;
    while (!cap.isOpened()) {
        cap.open(1, CAP_DSHOW); // 明确指定你的免驱云台相机索引
        if (!cap.isOpened()) {
            Sleep(800); // 没插上就一直等，不崩溃
        }
    }
    cout << "[Success] 外部相机已连接！" << endl;

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(CAP_PROP_FPS, 30);
    cap.set(CAP_PROP_BUFFERSIZE, 1);

    // 【曝光压低】关闭自动曝光，压暗画面
    // 由于加了偏振片且需要抵抗激光与高光干扰，这里重新开启并调低曝光
    cap.set(CAP_PROP_AUTO_EXPOSURE, 0.25); // 0.25 通常代表手动曝光模式 (不同电脑驱动可能不同，也可能为 0)
    cap.set(CAP_PROP_EXPOSURE, -6);        // 负值越小越暗，如果还是亮可以尝试 -7 或 -8

    // 【赛场热插拔机制：死等串口连接】
    cout << "[System] 正在等待插入串口线 (COM7)..." << endl;
    while (hSerial == INVALID_HANDLE_VALUE) {
        initSerial("COM7");
        if (hSerial == INVALID_HANDLE_VALUE) {
            Sleep(800);
        }
    }

    // 【新增】等待“开始瞄准”指令状态标识机
    bool is_aiming_started = false;
    cout << "\n============================================\n";
    cout << ">> 装备已全部连接！等待裁判指示要求开始..." << endl;
    cout << ">> 请按照规则，按 'S' 键发送“开始瞄准”指令" << endl;
    cout << "============================================\n";

    Mat frame;
    int target_color = -1;
    bool color_locked = false;
    Point last_valid_center = Point(-1, -1);
    int miss_frames = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 使用加入连续追踪逻辑的稳定中心点寻找
        Point center = detectCenterRobust(frame, last_valid_center, color_locked ? target_color : -1);

        // ===================== [修正：圆心稳定抗挂与防丢机制] =====================
        if (center.x >= 0) {
            if (last_valid_center.x < 0 || norm(center - last_valid_center) < 120) {
                last_valid_center = center;
                miss_frames = 0;
            }
            else {
                center = last_valid_center;
            }
        }
        else {
            miss_frames++;
            if (miss_frames < 30) {
                center = last_valid_center;
            }
            else {
                last_valid_center = Point(-1, -1);
                color_locked = false; // 大量丢失时重置状态
            }
        }

        if (center.x >= 0) {
            if (!color_locked) {
                target_color = detectCentralColor(frame, center);
                if (target_color >= 0) {
                    // 防呆：再次确认是不是杂色
                    color_locked = true;
                    cout << "Color locked: " << colorList[target_color].name << endl;
                }
            }
        }

        if (center.x >= 0 && color_locked) {
            Sector best;
            // 极大减少计算量，只寻找单一目标颜色
            bool found = detectTargetSector(frame, center, target_color, best);

            // =============== 【新增】角速度计算与超前预测防丢机制 ===============
            static double last_time = 0;
            static float last_angle = 0;
            static float smoothed_omega = 0;
            static float last_radius = 0;
            static int miss_target_frames = 0;

            double current_time = (double)getTickCount() / getTickFrequency();
            if (last_time == 0) last_time = current_time;
            double dt = current_time - last_time;

            Point predicted_st;

            if (found) {
                float test_angle = atan2(best.center.y - center.y, best.center.x - center.x);
                // 【核心滤波1：角度异常跳变剔除】
                // 能量转盘物理运动是连续的。若相邻帧发现目标角位移突变超 0.8 rad (约45度)，
                // 说明此色块极大概率是被卷入轨道的环境杂波灯牌，强制丢弃并交由底层盲推维持连贯平滑。
                if (miss_target_frames == 0 && last_time > 0) {
                    float d_a = test_angle - last_angle;
                    while (d_a > CV_PI) d_a -= 2 * CV_PI;
                    while (d_a < -CV_PI) d_a += 2 * CV_PI; 
                    if (abs(d_a) > 0.8f) {
                        found = false; 
                    }
                }
            }

            if (found) {
                miss_target_frames = 0;
                float current_angle = atan2(best.center.y - center.y, best.center.x - center.x);

                // 【核心滤波2：超高权重极坐标半径平滑】
                // 针对特征形变引起的质心上下抽搐：转盘扇叶几何形心到圆心物理距离应当绝对固定（~160mm）。
                // 利用极坐标思想固定极径 R——赋历史记录 85% 权重强行钳制抖动，将跳动完全抹平至无感！
                float current_radius = norm(best.center - center);
                if (last_radius == 0) last_radius = current_radius;
                last_radius = 0.85f * last_radius + 0.15f * current_radius;

                if (dt > 0.01 && dt < 0.5) {
                    float d_angle = current_angle - last_angle;
                    while (d_angle > CV_PI) d_angle -= 2 * CV_PI;
                    while (d_angle < -CV_PI) d_angle += 2 * CV_PI;

                    float omega = d_angle / dt;
                    if (omega > 2.5f) omega = 2.5f; // 前馈限幅：能量机关物理极限角速度设计值约 2.35 rad/s
                    if (omega < -2.5f) omega = -2.5f;

                    if (smoothed_omega == 0) smoothed_omega = omega;
                    // 【低延迟跟踪网络】
                    // 面对大符 (a*sin(wt)+b) 的高频非线性变速，过度滤波会导致严重相位延迟。
                    // 此处设定 80% 敏感度直接接纳当前观测角速度，缩短响应延迟。
                    smoothed_omega = 0.2f * smoothed_omega + 0.8f * omega;
                }
                last_angle = current_angle;
                last_time = current_time;

                float delay_sec = 0.25f;
                float predicted_angle = current_angle + smoothed_omega * delay_sec;

                predicted_st.x = center.x + last_radius * cos(predicted_angle);
                predicted_st.y = center.y + last_radius * sin(predicted_angle);
            }
            else {
                // 【绝影应对：开环预测盲推系统】
                miss_target_frames++;
                // 抵抗装甲板短暂越线暗区被遮挡或拉丝导致的掉帧问题。
                // 若短于 15 帧（约 0.5 秒）内发生视觉剥落，依靠最后稳定持有的惯性力学状态（角速度与极径），
                // 在没有图像支撑的背景中强行勾勒圆周轨迹，为串口持续续杯控制数据。
                if (miss_target_frames < 15 && last_radius > 0) {
                    float current_angle = last_angle + smoothed_omega * dt;
                    last_angle = current_angle; // 积分更新历史角度，给下一帧连续使用
                    last_time = current_time;

                    float delay_sec = 0.25f;
                    float predicted_angle = current_angle + smoothed_omega * delay_sec;

                    predicted_st.x = center.x + last_radius * cos(predicted_angle);
                    predicted_st.y = center.y + last_radius * sin(predicted_angle);

                    // 构造一个假的原目标原位点（供后续绿色线绘制不出错）
                    best.center.x = center.x + last_radius * cos(current_angle);
                    best.center.y = center.y + last_radius * sin(current_angle);

                    found = true; // 劫持为 true，使后续送串口控车生效！
                }
            }

            if (found) {
                // 【空间滞后消除机制】
                // 取消之前盲目的 predicted_st 加权平滑，防止预测提前角被加权平均向回拉扯。
                // 由于上方角速度 omega 已作滤波，此处直接应用无延迟的目标驱动坐标。
                Point st = predicted_st;
                float ax, ay;

                // 将目标坐标发送给计算函数
                calcAngles(st, ax, ay);
                // 【规则限制】：只有按下了“开始瞄准”才能控车发送数据
                if (is_aiming_started) {
                    sendAngles(ax, ay);
                }

                circle(frame, center, 5, { 0,255,0 }, -1);
                circle(frame, best.center, 6, { 0,0,255 }, -1); // 目标原来位置或盲推点
                circle(frame, st, 6, { 255,0,0 }, -1); // 目标预测打点位置 (蓝点)
                line(frame, Point(320, 240), st, { 255,255,0 }, 2);
                circle(frame, Point(320, 240), 3, { 255,0,0 }, -1);

                putText(frame, format("Target: %s %s [%s]", colorList[target_color].name.c_str(),
                    is_aiming_started ? "[AIMING!]" : "[WAIT]", miss_target_frames > 0 ? "TRACKING" : "LOCKED"),
                    { 10,30 }, FONT_HERSHEY_SIMPLEX, 0.7, is_aiming_started ? Scalar(0, 0, 255) : Scalar(0, 255, 0), 2);
                putText(frame, format("ErrX:%.1f ErrY:%.1f", ax, ay),
                    { 10,60 }, 0, 0.6, { 0,255,0 }, 2);
            }
            else {
                putText(frame, "TARGET SECTOR NOT FOUND", { 10,60 }, 0, 0.7, { 0,0,255 }, 2);
            }
        }
        else {
            putText(frame, "CENTER LOST, RESEARCHING...", { 10,30 }, 0, 0.7, { 0,0,255 }, 2);
        }

        imshow("Tracking", frame);

        // 键盘监听
        int key = waitKey(1);
        if (key == 27) break; // ESC 退出
        if (key == 's' || key == 'S') {
            if (!is_aiming_started) {
                is_aiming_started = true;
                cout << "=== [状态切换]：开始瞄准！已开启激光！ ===" << endl;
                // 此时给下位机发指令 '#V!'（由于STM32单片机端只认#和!包裹的数据包）
                DWORD bw;
                if (hSerial != INVALID_HANDLE_VALUE) WriteFile(hSerial, "#V!", 3, &bw, NULL);
            }
        }
        if (key == 'r' || key == 'R') {
            color_locked = false;
            target_color = -1;
            last_valid_center = Point(-1, -1);
            cout << "=== [状态切换]：清空记忆，重新瞄准！ ===" << endl;
        }
    }

    // 断开连接前发送 '#Q!' 指令给下位机，下位机同样需要带上包头和包尾才能被成功解析
    if (hSerial != INVALID_HANDLE_VALUE) {
        DWORD bw;
        WriteFile(hSerial, "#Q!", 3, &bw, NULL);
        cout << "=== [下线]已向单片机发送断开(关闭激光)指令===" << endl;
    }

    cap.release();
    destroyAllWindows();
    if (hSerial != INVALID_HANDLE_VALUE) CloseHandle(hSerial);
    return 0;
}
