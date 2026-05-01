import requests
import json
import sys
import threading
import os

# 警告，该代码请谨慎运行，多线程并发可能产生大量费用，请确保您的账号有足够的余额

# 用前必读
# 1. 请将模型名称替换为您需要使用的模型名称
# 2. 请将API密钥替换为您在系统中复制的APIKEY(令牌)
# 3. 代码中已支持流式接收响应，不建议修改为非流式，否则会导致响应时间过长
# 本代码为示例代码，具体使用请根据实际情况修改

# 定义模型和API配置
model = "deepseek-chat"
api_key = "sk-KQ9WCl390A6NujZAs7k0EHCgm44HdsXgJ5YuDEGNCIZO63p7"

# 警告，该代码请谨慎运行，多线程并发可能产生大量费用，请确保您的账号有足够的余额
# 支持通过参数自定义并发线程数与总请求数。
# 每个线程完成后一次性输出自己的结果，使用 [T<编号>] 前缀。
# 线程数, 当前表示不并发 1 条线程执行, 谨慎调整线程数, 目前不同模型的默认RPM限制在几千到几万不等，如果你的请求量非常大, 请提前告知客服, 以避免请求被拦截或账户被封禁
THREADS = 1
# 总请求数, 当前表示所有线程总计请求 20 次后程序停止执行
REQUESTS = 200
MESSAGE = "你好"

# 定义请求头
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 全局 Session + 连接池，用于复用连接，避免高并发时频繁建立连接
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=THREADS * 2 or 10,
    pool_maxsize=THREADS * 2 or 10,
)
session.mount("http://", adapter)
session.mount("https://", adapter)

print_lock = threading.Lock()

sys.stdout.reconfigure(encoding="utf-8")

URL = "https://api.silra.cn/v1/chat/completions"


def deepseek(
    message: str,
    *,
    url: str = None,
    model_name: str = None,
    api_key_override: str = None,
    timeout_s: int = 600,
) -> str:
    m = (message or "").strip()
    if not m:
        return ""
    req_url = url or os.getenv("DEEPSEEK_API_URL") or URL
    req_model = model_name or os.getenv("DEEPSEEK_MODEL") or model
    req_key = api_key_override or os.getenv("DEEPSEEK_API_KEY") or api_key
    req_headers = {
        "Authorization": f"Bearer {req_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": req_model,
        "messages": [{"role": "user", "content": m}],
        "stream": True,
    }
    response = session.post(
        req_url,
        headers=req_headers,
        json=data,
        timeout=int(timeout_s),
        stream=True,
    )
    response.raise_for_status()
    response.encoding = "utf-8"
    buf = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        payload = line[6:] if line.startswith("data: ") else line
        if payload.strip() == "[DONE]":
            break
        try:
            obj = json.loads(payload)
            choices = obj.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                token = delta.get("content")
                if token is None:
                    message_obj = choices[0].get("message", {})
                    token = message_obj.get("content")
                if token:
                    buf.append(token)
        except Exception:
            pass
    return "".join(buf)


def call(req_id: int, url: str, message: str):
    data = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": True
    }
    try:
        # 使用全局 session 进行请求，复用底层连接
        response = session.post(
            url,
            headers=headers,
            json=data,
            timeout=600,
            stream=True
        )
        if response.status_code == 200:
            response.encoding = "utf-8"
            buf = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                payload = line[6:] if line.startswith("data: ") else line
                if payload.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                    choices = obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        token = delta.get("content")
                        if token is None:
                            message_obj = choices[0].get("message", {})
                            token = message_obj.get("content")
                        if token:
                            buf.append(token)
                except Exception:
                    pass
            text = "".join(buf)
            with print_lock:
                print(f"[T{req_id}] {text}", flush=True)
            return True
        else:
            with print_lock:
                print(f"[T{req_id}] 请求失败 {response.status_code}: {response.text}", flush=True)
            return False
    except requests.exceptions.Timeout:
        with print_lock:
            print(f"[T{req_id}] 请求超时", flush=True)
        return False
    except requests.exceptions.RequestException as e:
        with print_lock:
            print(f"[T{req_id}] 请求发生错误: {e}", flush=True)
        return False


# 警告，该代码请谨慎运行，多线程并发可能产生大量费用，请确保您的账号有足够的余额
def main():
    total = REQUESTS if REQUESTS is not None else THREADS
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = [ex.submit(call, i + 1, URL, MESSAGE) for i in range(total)]
        for _ in as_completed(futures):
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        # 程序结束时关闭 Session，释放连接资源
        session.close()
