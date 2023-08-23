import subprocess
import psutil

def get_gpu_processes():
    try:
        # 执行 nvidia-smi 命令，并捕获输出
        output = subprocess.check_output(['nvidia-smi', '-q', '-x'])
        output = output.decode('utf-8')

        # 解析 XML 输出
        import xml.etree.ElementTree as ET
        root = ET.fromstring(output)

        # 获取所有 GPU 进程的信息
        gpu_processes = []
        for gpu in root.findall('gpu'):
            gpu_processes.extend(gpu.findall('.//process_info'))

        # 提取进程的 PID
        pids = [process.find('pid').text for process in gpu_processes]

        return pids

    except subprocess.CalledProcessError:
        print('Failed to execute nvidia-smi command.')

def get_process_username(pid):
    try:
        # 使用PID获取进程详情
        process = psutil.Process(pid)
        
        # 获取进程的用户名
        username = process.username()
        
        return username
    
    except psutil.Error:
        print('Failed to retrieve process information.')


# 调用函数获取显卡上的进程 PID 列表
pids = get_gpu_processes()
for pid in pids:
    # 根据 PID 获取进程的用户名
    username = get_process_username(int(pid))
    print(f'PID: {pid}, Username: {username}')