"""
短临预报 Demo 后端
零依赖：只用 Python 标准库 http.server + json
启动：python server.py  →  浏览器打开 http://localhost:8080
"""
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================
# LLM 客户端 & 数据 & 提示词（从 notebook 提取）
# ============================================================

llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)
LLM_MODEL = "deepseek-chat"


def build_forecast_data_text():
    return """【雷达数据分析结果】
- 时次: 2026-06-16 12:00
- 数据来源: 雷达实况 1个文件, 雷达外推 1个文件
- 总时间跨度: 1小时 (当前雷达实况 + 未来1-2小时外推)

【一、雷达实况分析】
- 深圳内对流单体信息
  - 识别到对流单体: 0个
- 深圳实际反射率覆盖区域统计
  基于实际反射率（18dBZ以上）分析全市及各区（含深汕）覆盖率，输出信息包括：区域名称、覆盖率、强度。
  - 深圳（全市）:覆盖率:90.3%, 最大反射率:47dBZ, 平均反射率:32dBZ, 强度:强对流
  - 福田:覆盖率:64.8%, 最大反射率:43dBZ, 平均反射率:38dBZ, 强度:强
  - 光明:覆盖率:57.1%, 最大反射率:47dBZ, 平均反射率:38dBZ, 强度:强对流
  - 龙华:覆盖率:50.7%, 最大反射率:44dBZ, 平均反射率:29dBZ, 强度:强
  - 大鹏:覆盖率:50.3%, 最大反射率:34dBZ, 平均反射率:25dBZ, 强度:中等
  - 罗湖:覆盖率:49.3%, 最大反射率:42dBZ, 平均反射率:36dBZ, 强度:强
  - 坪山:覆盖率:48.6%, 最大反射率:37dBZ, 平均反射率:30dBZ, 强度:强
  - 盐田:覆盖率:40.6%, 最大反射率:39dBZ, 平均反射率:35dBZ, 强度:强
  - 宝安:覆盖率:37.3%, 最大反射率:46dBZ, 平均反射率:29dBZ, 强度:强对流
  - 龙岗:覆盖率:30.2%, 最大反射率:44dBZ, 平均反射率:29dBZ, 强度:强
  - 南山:覆盖率:27.7%, 最大反射率:42dBZ, 平均反射率:35dBZ, 强度:强
  - 深汕:覆盖率:96.8%, 最大反射率:46dBZ, 平均反射率:34dBZ, 强度:强对流
- 深圳周边对流单体信息
  - 识别到对流单体: 20个
  - ID:150,区域:珠江口,强度:58.0dBZ
  - ID:158,区域:珠江口,强度:57.0dBZ
  - ID:6,区域:珠江口,强度:55.0dBZ
  - ID:1,区域:珠江口,强度:52.0dBZ
  - ID:11,区域:香港,强度:51.0dBZ
  - ID:236,区域:珠江口,强度:51.0dBZ
  - ID:64,区域:珠江口,强度:50.0dBZ
  - ID:164,区域:珠江口,强度:50.0dBZ
  - ID:21,区域:珠江口,强度:49.0dBZ
  - ID:96,区域:珠江口,强度:48.0dBZ
  - ID:36,区域:珠江口,强度:47.0dBZ
  - ID:192,区域:珠江口,强度:47.0dBZ
  - ID:179,区域:珠江口,强度:45.0dBZ
  - ID:386,区域:珠江口,强度:45.0dBZ
  - ID:15,区域:珠江口,强度:44.0dBZ
  - ID:110,区域:珠江口,强度:44.0dBZ
  - ID:122,区域:珠江口,强度:44.0dBZ
  - ID:34,区域:珠江口,强度:42.0dBZ
  - ID:40,区域:珠江口,强度:42.0dBZ
  - ID:59,区域:珠江口,强度:42.0dBZ

【二、雷达外推与降雨分析】
- 雷达外推分析
  - 强度分析: 反射率变化 0.0dBZ
  - 移动分析: 向东南缓慢移动, 速度 20 km/h
  - 影响覆盖面分析: 覆盖面变化 0.0%
  - 云团形态识别: 块状强回波, 强对流天气，可能有雷暴
【降雨预报分析】
  -【未来2小时降雨强度趋势分析】(12:06-14:00)
   - 全市平均雨量变化时序（mm）:2.5, 2.4, 2.4, 2.2, 2.3, 2.4, 2.5, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.6, 2.6, 2.6, 2.6, 2.5, 2.4
  -【未来1小时降雨预报】(13:00)
   - 深圳（全市）:覆盖100.1%, 最大67.8毫米, 最小0.1毫米, 平均8.4毫米, 大暴雨
   - 光明: 覆盖100.0%, 最大67.8毫米, 最小2.9毫米, 平均44.3毫米, 暴雨
   - 宝安: 覆盖98.9%, 最大62.6毫米, 最小0.6毫米, 平均17.4毫米, 暴雨
   - 龙华: 覆盖97.4%, 最大58.7毫米, 最小4.9毫米, 平均30.7毫米, 暴雨
   - 龙岗: 覆盖100.0%, 最大30.7毫米, 最小0.1毫米, 平均4.5毫米, 暴雨
   - 南山: 覆盖100.0%, 最大20.8毫米, 最小0.1毫米, 平均5.6毫米, 大雨
   - 福田: 覆盖100.0%, 最大6.2毫米, 最小4.3毫米, 平均5.0毫米, 中雨
   - 罗湖: 覆盖100.0%, 最大5.8毫米, 最小5.0毫米, 平均5.4毫米, 中雨
   - 盐田: 覆盖100.0%, 最大5.6毫米, 最小4.7毫米, 平均5.0毫米, 中雨
   - 坪山: 覆盖100.0%, 最大4.8毫米, 最小2.5毫米, 平均4.0毫米, 小雨
   - 深汕: 覆盖100.0%, 最大4.1毫米, 最小0.2毫米, 平均1.8毫米, 小雨
   - 大鹏: 覆盖98.8%, 最大2.2毫米, 最小0.1毫米, 平均0.5毫米, 小雨
  -【未来2小时降雨预报】(14:00)
   - 深圳（全市）:覆盖100.1%, 最大44.9毫米, 最小0.4毫米, 平均6.2毫米, 暴雨
   - 光明: 覆盖100.0%, 最大44.9毫米, 最小4.2毫米, 平均30.3毫米, 暴雨
   - 龙华: 覆盖97.4%, 最大40.4毫米, 最小5.7毫米, 平均20.5毫米, 暴雨
   - 宝安: 覆盖98.9%, 最大36.1毫米, 最小2.2毫米, 平均10.2毫米, 暴雨
   - 龙岗: 覆盖100.0%, 最大13.0毫米, 最小0.9毫米, 平均4.5毫米, 中雨
   - 南山: 覆盖100.0%, 最大7.9毫米, 最小0.8毫米, 平均4.9毫米, 中雨
   - 福田: 覆盖100.0%, 最大6.8毫米, 最小5.0毫米, 平均6.0毫米, 中雨
   - 罗湖: 覆盖100.0%, 最大6.5毫米, 最小5.7毫米, 平均6.0毫米, 中雨
   - 盐田: 覆盖100.0%, 最大5.9毫米, 最小5.4毫米, 平均5.7毫米, 中雨
   - 坪山: 覆盖100.0%, 最大5.9毫米, 最小4.0毫米, 平均5.1毫米, 中雨
   - 大鹏: 覆盖98.8%, 最大4.4毫米, 最小0.9毫米, 平均2.4毫米, 小雨
   - 深汕: 覆盖100.0%, 最大2.7毫米, 最小0.4毫米, 平均1.0毫米, 小雨
"""

SYSTEM_PROMPT = """你是深圳市气象台的预报员，请根据下面的数据分析数据，写一段面向决策服务的短临服务提示。内容要求如下：
## 核心规则
1. 使用通俗易懂的语言，不失气象专业与严谨，结构清晰，重点突出，简单明了。只需要一段话。
2. 关于雷达回波的描述：以深圳为中心，只需要一句话概括。不要具体数字，以文字描述为主。不需要反射率数据，不需要移动速度数据等。
3. 当深圳已出现强回波时，简单研判是否会加强；反之，则监测深圳周边的雷达回波，简单研判多久会影响深圳。
4. 内容构成：
   - 第一句话：总体分析（根据全部数据一句话描述全市天气形势，不要用时效类话术）。
   - 第二句：降雨预报（以"预计未来x-x小时"开头，可按总体-重点（达到暴雨的）-次要的顺序分析总结；可按降雨级别、行政区域进行归类分析；逻辑严谨，降雨级别不能改，跨级别描述，小的放前面）。
5. 把未来天气情况放到一起，不要重复使用"未来"、"预计"等字样（除第二句必需的"预计未来x-x小时"开头外）。
6. 按区域描述的不带具体行政区名称（用"中西部""南部""北部""东部""西部"等大区名称代替）。
7. 雨量数据不能自行估测，降雨量数值规范：直接去掉小数部分，精确到个位，个位接近1的个位取1，个位接近5的个位取5，接近10的个位取0十位进1，区间值中间用"—"隔开。
8. 区域划分：中部(罗湖福田)、西部(宝安光明南山龙华)、东部(盐田坪山大鹏)、南部(南山福田罗湖盐田大鹏)、北部(光明龙华龙岗坪山)、中西部(罗湖福田宝安光明南山龙华)、深汕。
9. 降雨等级标准：中雨(5.0≤R≤14.9)、大雨(15.0≤R≤29.9)、暴雨(30.0≤R)、阵雨/小雨(0.1≤R≤4.9)。
10. 各区域不能重复出现。
11. 重复生成时，描述顺序与风格尽可能保持一致。

## 多轮对话
用户可能会对生成的预报文本提出修改意见，你需要根据意见修改文本，同时保持其他内容不变。常见请求：
- "简单点"/"精简一点" → 缩短文本，保留核心信息
- "把预报雨量调大点" → 在合理范围内将雨量值略上调
- "暴雨重点说" → 突出暴雨区域描述
- "换一种说法" → 保持内容但调整表达方式

请只返回最终的预报段落文本，不要加引号或额外解释。"""


# ============================================================
# ForecastChat 对话管理
# ============================================================

class ForecastChat:
    def __init__(self, system_prompt=SYSTEM_PROMPT, model=LLM_MODEL):
        self.model = model
        self.messages = [{"role": "system", "content": system_prompt}]
        self.current_forecast = None

    def _call_llm(self):
        try:
            response = llm_client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=0.2,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLM调用失败] {e}"

    def generate(self, forecast_data_text):
        user_msg = f"请根据以下数据，按规则写一段短临服务提示：\n\n{forecast_data_text}"
        self.messages.append({"role": "user", "content": user_msg})
        reply = self._call_llm()
        self.messages.append({"role": "assistant", "content": reply})
        self.current_forecast = reply
        return reply

    def refine(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        reply = self._call_llm()
        self.messages.append({"role": "assistant", "content": reply})
        self.current_forecast = reply
        return reply

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.current_forecast = None

    def get_current(self):
        return self.current_forecast


# ============================================================
# 全局对话实例
# ============================================================

chat = ForecastChat()
data_text = build_forecast_data_text()

# ============================================================
# HTTP API（纯 stdlib，无依赖）
# ============================================================

class APIHandler(SimpleHTTPRequestHandler):
    """处理 API 请求 + 静态文件"""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_file("index.html", "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._json_response({"current": chat.get_current()})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/generate":
            forecast = chat.generate(data_text)
            self._json_response({"forecast": forecast})
        elif self.path == "/api/refine":
            body = self._read_body()
            user_input = body.get("input", "")
            result = chat.refine(user_input)
            self._json_response({"forecast": result})
        elif self.path == "/api/reset":
            chat.reset()
            self._json_response({"ok": True})
        else:
            self._json_response({"error": "not found"}, 404)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filename, content_type):
        filepath = Path(__file__).parent / filename
        if not filepath.exists():
            self.send_error(404, f"File not found: {filename}")
            return
        content = filepath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        # 精简日志输出
        print(f"[{self.command}] {args[0]}")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(("0.0.0.0", port), APIHandler)
    print(f"服务已启动 → http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已关闭")
        server.server_close()
