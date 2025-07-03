#include "serial.hpp"

#include <string>

using namespace toe;

void serial::init_port(const nlohmann::json & input)
{
    post_path = input["path"]["serial_path"].get<std::string>();
    init_port();
}

void serial::init_port()
{
    while (1)
    {
        serial_fd.open(post_path, m_ec);
        if (m_ec)
        {
            std::cout << "打开串口失败，错误信息:" << m_ec.message() << std::endl;
            continue;
        }
        //设置串口参数  
        // 串口参数设置
        // 波特率 115200
        serial_fd.set_option(boost::asio::serial_port::baud_rate(115200), m_ec);
        if (m_ec)
        {
            std::cout << "串口参数设置失败，错误信息:" << m_ec.message() << std::endl;
            continue;
        }
        // 流控制 none
        serial_fd.set_option(boost::asio::serial_port::flow_control(boost::asio::serial_port::flow_control::none), m_ec);  
        if (m_ec)
        {
            std::cout << "串口参数设置失败，错误信息:" << m_ec.message() << std::endl;
            continue;
        }
        //奇偶校验 none
        serial_fd.set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none), m_ec);  
        if (m_ec)
        {
            std::cout << "串口参数设置失败，错误信息:" << m_ec.message() << std::endl;
            continue;
        }
        // 停止位 one
        serial_fd.set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one), m_ec);  
        if (m_ec)
        {
            std::cout << "串口参数设置失败，错误信息:" << m_ec.message() << std::endl;
            continue;
        }
        // 字符大小 8
        serial_fd.set_option(boost::asio::serial_port::character_size(8), m_ec);
        if (m_ec)
        {
            std::cout << "串口参数设置失败，错误信息:"<< m_ec.message() << std::endl;
            continue;
        }
        break;
    }
}

void serial::get_msg(int & color, int & mode)
{
    input_mutex_.lock();
    memset(&rbuff,0,sizeof(rbuff));
    serial_fd.read_some(boost::asio::buffer(rbuff), m_ec);  
    input_mutex_.unlock();
    if (!m_ec)
    {
        // 校验帧头
        if (int(rbuff[0]) == 166)
        {
            size_t start_index = 1;
            // color1为蓝色,0为红色
            if (int(rbuff[start_index++]) > 100)
            {
                color = true;
            }
            else
            {
                color = false;
            }
            // 模式信息:0为自瞄,1为参数整定
            mode = int(rbuff[start_index++]);
        }
    }
}

// 将 std::vector<double> 中的数据打包成 char[] 并发送到串口
void serial::send_msg(const std::vector<double> & msg)
{
    if (msg.size() >= 3)
    {
        output_mutex_.lock();

        acm_data temp;
        temp.data = (float)msg[0];
        sbuff[0] = temp.bit[0];
        sbuff[1] = temp.bit[1];
        sbuff[2] = temp.bit[2];
        sbuff[3] = temp.bit[3];
        temp.data = (float)msg[1];
        sbuff[4] = temp.bit[0];
        sbuff[5] = temp.bit[1];
        sbuff[6] = temp.bit[2];
        sbuff[7] = temp.bit[3];
        temp.data = (float)msg[2];
        sbuff[8] = temp.bit[0];
        sbuff[9] = temp.bit[1];
        sbuff[10] = temp.bit[2];
        sbuff[11] = temp.bit[3];

        serial_fd.write_some(boost::asio::buffer(sbuff) ,m_ec);  
        output_mutex_.unlock();
    }
}

// 发送ASCII字符串
void serial::send_msg(const char* data, size_t len)
{
    output_mutex_.lock();
    boost::asio::write(serial_fd, boost::asio::buffer(data, len), m_ec);
    output_mutex_.unlock();
}