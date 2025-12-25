import base64
import os
import uuid
import httpx

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, register


@register("AstrBot_Plugins_Canvas", "stevessr", "使用 Gemini 图像模型（支持 gemini-3-pro, gemini-2.5-flash-image 等）进行图片内容创作的插件", "v1.2")
class GeminiImageGenerator(Star):
    # 支持的模型列表
    SUPPORTED_MODELS = [
        "gemini-3-pro",
        "gemini-2.5-flash-image"
    ]

    # 支持的图片比例
    SUPPORTED_ASPECT_RATIOS = [
        "1:1",    # 正方形
        "16:9",   # 横向宽屏
        "9:16",   # 竖向
        "4:3",    # 传统横向
        "3:4",    # 传统竖向
        "21:9",   # 超宽屏
        "9:21",   # 超竖屏
        "5:4",    # 接近正方形横向
        "4:5",    # 接近正方形竖向
        "3:2",    # 经典照片横向
        "2:3",    # 经典照片竖向
    ]

    # 比例别名映射（方便用户使用）
    ASPECT_RATIO_ALIASES = {
        "正方形": "1:1",
        "方形": "1:1",
        "横屏": "16:9",
        "宽屏": "16:9",
        "竖屏": "9:16",
        "竖向": "9:16",
        "横向": "4:3",
        "超宽": "21:9",
        "超宽屏": "21:9",
        "超竖": "9:21",
        "照片": "3:2",
        "照片横": "3:2",
        "照片竖": "2:3",
    }

    # 支持的分辨率
    SUPPORTED_RESOLUTIONS = {
        "1k": 1024,
        "2k": 2048,
        "4k": 4096,
    }

    # 分辨率别名
    RESOLUTION_ALIASES = {
        "低": "1k",
        "中": "2k",
        "高": "4k",
        "low": "1k",
        "medium": "2k",
        "high": "4k",
    }

    def __init__(self, context: Context, config: AstrBotConfig):
        """Gemini 图片生成与编辑插件初始化"""
        super().__init__(context)
        self.config = config

        logger.info(f"插件配置加载成功：{self.config}")

        # 读取多密钥配置
        self.api_keys = self.config.get("gemini_api_keys", [])
        self.current_key_index = 0

        # 读取模型配置（默认使用 gemini-3-pro）
        self.model_name = self.config.get("model_name", "gemini-3-pro")
        logger.info(f"使用模型：{self.model_name}")

        # 读取默认图片比例配置
        self.default_aspect_ratio = self.config.get("aspect_ratio", "1:1")
        if self.default_aspect_ratio not in self.SUPPORTED_ASPECT_RATIOS:
            logger.warning(f"不支持的比例 {self.default_aspect_ratio}，使用默认 1:1")
            self.default_aspect_ratio = "1:1"
        logger.info(f"默认图片比例：{self.default_aspect_ratio}")

        # 读取默认分辨率配置
        self.default_resolution = self.config.get("resolution", "1k").lower()
        if self.default_resolution not in self.SUPPORTED_RESOLUTIONS:
            logger.warning(f"不支持的分辨率 {self.default_resolution}，使用默认 1k")
            self.default_resolution = "1k"
        logger.info(f"默认分辨率：{self.default_resolution}")

        # 是否显示模型的文本输出（用于提高生成效果）
        self.show_model_text = self.config.get("show_model_text", True)
        logger.info(f"显示模型文本输出：{self.show_model_text}")

        # 初始化图片保存目录
        plugin_dir = os.path.dirname(__file__)
        self.save_dir = os.path.join(plugin_dir, "temp_images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            logger.info(f"已创建图片临时目录：{self.save_dir}")

        # 初始化配置
        self.api_base_url = self.config.get(
            "api_base_url", "https://generativelanguage.googleapis.com"
        )

        if not self.api_keys:
            logger.error("未配置任何 Gemini API 密钥，请在插件配置中填写")

    def _get_current_api_key(self):
        """获取当前使用的 API 密钥"""
        if not self.api_keys:
            return None
        return self.api_keys[self.current_key_index]

    def _switch_next_api_key(self):
        """切换到下一个 API 密钥"""
        if not self.api_keys:
            return
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.info(f"已切换到下一个 API 密钥（索引：{self.current_key_index}）")

    def _parse_generation_params(self, text: str) -> tuple[str, str, str]:
        """从文本中解析生成参数

        支持格式：
        - --ratio 16:9 或 -r 16:9 或 --比例 横屏
        - --res 2k 或 --resolution 4k 或 --分辨率 高

        Returns:
            (清理后的提示词，比例，分辨率)
        """
        import re

        aspect_ratio = self.default_aspect_ratio
        resolution = self.default_resolution
        clean_text = text

        # 匹配比例参数
        ratio_patterns = [
            r'(?:--ratio|-r)\s+(\S+)',
            r'--比例\s+(\S+)',
        ]

        for pattern in ratio_patterns:
            match = re.search(pattern, clean_text)
            if match:
                ratio_input = match.group(1)
                if ratio_input in self.ASPECT_RATIO_ALIASES:
                    aspect_ratio = self.ASPECT_RATIO_ALIASES[ratio_input]
                elif ratio_input in self.SUPPORTED_ASPECT_RATIOS:
                    aspect_ratio = ratio_input
                else:
                    logger.warning(f"不支持的比例参数：{ratio_input}，使用默认 {self.default_aspect_ratio}")
                clean_text = re.sub(pattern, '', clean_text).strip()
                break

        # 匹配分辨率参数
        res_patterns = [
            r'(?:--res|--resolution)\s+(\S+)',
            r'--分辨率\s+(\S+)',
        ]

        for pattern in res_patterns:
            match = re.search(pattern, clean_text)
            if match:
                res_input = match.group(1).lower()
                if res_input in self.RESOLUTION_ALIASES:
                    resolution = self.RESOLUTION_ALIASES[res_input]
                elif res_input in self.SUPPORTED_RESOLUTIONS:
                    resolution = res_input
                else:
                    logger.warning(f"不支持的分辨率参数：{res_input}，使用默认 {self.default_resolution}")
                clean_text = re.sub(pattern, '', clean_text).strip()
                break

        return clean_text, aspect_ratio, resolution

    @filter.command("gemini_image", alias={"文生图"})
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        """根据文本描述生成图片

        支持可选参数：
        - 比例：--ratio 16:9 或 -r 横屏 或 --比例 竖屏
        - 分辨率：--res 2k 或 --resolution 4k 或 --分辨率 高
        示例：/gemini_image 一只猫 --ratio 16:9 --res 2k
        """
        if not self.api_keys:
            yield event.plain_result("错误：未配置任何 Gemini API 密钥")
            return

        if not prompt.strip():
            yield event.plain_result(
                "请输入图片描述\n"
                "示例：/gemini_image 一只戴帽子的猫在月球上\n"
                "可选比例：--ratio 16:9 或 -r 横屏\n"
                "可选分辨率：--res 2k 或 --分辨率 高\n"
                f"支持的比例：{', '.join(self.SUPPORTED_ASPECT_RATIOS)}\n"
                f"支持的分辨率：{', '.join(self.SUPPORTED_RESOLUTIONS.keys())}"
            )
            return

        # 解析生成参数
        clean_prompt, aspect_ratio, resolution = self._parse_generation_params(prompt)

        save_path = None

        try:
            yield event.plain_result(f"正在生成图片（比例：{aspect_ratio}，分辨率：{resolution}），请稍等...")
            result = await self._generate_image_with_retry(clean_prompt, aspect_ratio, resolution)

            if not result:
                logger.error("生成失败：所有 API 密钥均尝试完毕")
                yield event.plain_result("生成失败：所有 API 密钥均尝试失败")
                return

            image_data, model_text = result

            # 保存图片
            file_name = f"{uuid.uuid4()}.png"
            save_path = os.path.join(self.save_dir, file_name)

            with open(save_path, "wb") as f:
                f.write(image_data)

            logger.info(f"图片已保存至：{save_path}")

            # 构建发送内容
            result_chain = [Image.fromFileSystem(save_path)]

            # 如果有模型文本输出且配置允许显示，则添加文本
            if model_text and self.show_model_text:
                result_chain.append(Comp.Plain(f"\n\n{model_text}"))

            # 发送图片和文本
            yield event.chain_result(result_chain)
            logger.info(f"图片发送成功，提示词：{clean_prompt}，比例：{aspect_ratio}，分辨率：{resolution}")

        except Exception as e:
            logger.error(f"图片处理失败：{str(e)}")
            yield event.plain_result(f"生成失败：{str(e)}")

        finally:
            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info(f"已删除临时图片：{save_path}")
                except Exception as e:
                    logger.warning(f"删除临时图片失败：{str(e)}")

    @filter.command("gemini_edit", alias={"图编辑"})
    async def edit_image(self, event: AstrMessageEvent, prompt: str):
        """仅支持：引用图片后发送指令编辑图片"""
        if not self.api_keys:
            yield event.plain_result("错误：未配置任何 Gemini API 密钥")
            return

        # 图片提取逻辑
        image_path = await self._extract_image_from_reply(event)
        if not image_path:
            yield event.plain_result("未找到图片，请先长按图片发送回复后重试")
            return

        # 图片编辑处理
        async for result in self._process_image_edit(event, prompt, image_path):
            yield result

    @filter.llm_tool(name="edit_image")
    async def edit_image_tool(self, event: AstrMessageEvent, prompt: str):
        """编辑现有图片。当你需要编辑图片时，请使用此工具。

        Args:
            prompt(string): 编辑描述（例如：把猫咪改成黑色）
        """
        if not self.api_keys:
            yield event.plain_result("错误：未配置任何 Gemini API 密钥")
            return

        if not prompt.strip():
            yield event.plain_result("请提供编辑描述（例如：把猫咪改成黑色）")
            return

        image_path = await self._extract_image_from_reply(event)
        if not image_path:
            yield event.plain_result(
                "未找到图片，请先长按图片并点击“回复”，再输入编辑指令"
            )
            return

        async for result in self._process_image_edit(event, prompt, image_path):
            yield result

    @filter.llm_tool(name="generate_image")
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str, aspect_ratio: str = "", resolution: str = ""):
        """根据文本描述生成图片，当你需要生成图片时请使用此工具。

        Args:
            prompt(string): 图片描述文本（例如：画只猫）
            aspect_ratio(string): 可选，图片比例（1:1/16:9/9:16/4:3/3:4/21:9/5:4/4:5/3:2/2:3），默认为配置值
            resolution(string): 可选，分辨率（1k/2k/4k），默认为配置值
        """
        # 如果 LLM 指定了比例或分辨率，添加到 prompt 中
        if aspect_ratio and aspect_ratio in self.SUPPORTED_ASPECT_RATIOS:
            prompt = f"{prompt} --ratio {aspect_ratio}"
        if resolution and resolution.lower() in self.SUPPORTED_RESOLUTIONS:
            prompt = f"{prompt} --res {resolution}"
        async for result in self.generate_image(event, prompt):
            yield result

    # 提取回复中图片
    async def _extract_image_from_reply(self, event: AstrMessageEvent):
        """从回复消息中提取图片并返回本地路径"""
        try:
            message_components = event.message_obj.message
            reply_component = None
            for comp in message_components:
                if isinstance(comp, Comp.Reply):
                    reply_component = comp
                    logger.info(f"检测到回复消息（ID：{comp.id}），提取被引用图片")
                    break

            if not reply_component:
                logger.warning("未检测到回复组件（用户未长按图片回复）")
                return None

            # 从回复的 chain 中提取 Image 组件
            image_component = None
            for quoted_comp in reply_component.chain:
                if isinstance(quoted_comp, Comp.Image):
                    image_component = quoted_comp
                    logger.info(
                        f"从回复中提取到图片组件（file：{image_component.file}）"
                    )
                    break

            if not image_component:
                logger.warning("回复中未包含图片组件")
                return None

            # 获取本地图片路径（自动处理下载/转换）
            image_path = await image_component.convert_to_file_path()
            logger.info(f"图片已处理为本地路径：{image_path}")
            return image_path

        except Exception as e:
            logger.error(f"提取图片失败：{str(e)}", exc_info=True)
            return None

    # 统一的图片编辑处理逻辑
    async def _process_image_edit(
        self, event: AstrMessageEvent, prompt: str, image_path: str
    ):
        """处理图片编辑的核心逻辑"""
        save_path = None
        try:
            yield event.plain_result("正在编辑图片，请稍等...")

            # 调用带重试的编辑方法
            image_data = await self._edit_image_with_retry(prompt, image_path)

            if not image_data:
                yield event.plain_result("编辑失败：所有 API 密钥均尝试失败")
                return

            # 保存并发送编辑后的图片
            save_path = os.path.join(self.save_dir, f"{uuid.uuid4()}_edited.png")
            with open(save_path, "wb") as f:
                f.write(image_data)

            yield event.chain_result([Comp.Image.fromFileSystem(save_path)])
            logger.info(f"图片编辑完成并发送，提示词：{prompt}")

        except Exception as e:
            logger.error(f"图片编辑出错：{str(e)}")
            yield event.plain_result(f"图片编辑失败：{str(e)}")

        finally:
            # 清理临时文件
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    logger.info(f"已删除原始图片临时文件：{image_path}")
                except Exception as e:
                    logger.warning(f"删除原始图片失败：{str(e)}")

            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info(f"已删除编辑图临时文件：{save_path}")
                except Exception as e:
                    logger.warning(f"删除编辑图失败：{str(e)}")

    async def _retry_with_fallback_keys(self, operation_name: str, operation_func, *args, **kwargs):
        """通用的 API 密钥重试逻辑

        Args:
            operation_name: 操作名称（用于日志）
            operation_func: 要执行的操作函数（异步函数）
            *args, **kwargs: 传递给操作函数的参数

        Returns:
            操作成功时返回函数结果，失败时返回 None
        """
        max_attempts = len(self.api_keys)
        attempts = 0

        while attempts < max_attempts:
            current_key = self._get_current_api_key()
            if not current_key:
                break

            logger.info(
                f"尝试{operation_name}（密钥索引：{self.current_key_index}，尝试次数：{attempts + 1}/{max_attempts}）"
            )

            try:
                return await operation_func(current_key, *args, **kwargs)
            except Exception as e:
                attempts += 1
                logger.error(f"第{attempts}次尝试失败：{str(e)}")
                if attempts < max_attempts:
                    self._switch_next_api_key()
                else:
                    logger.error("所有 API 密钥均尝试失败")

        return None

    async def _edit_image_with_retry(self, prompt, image_path):
        """带重试逻辑的图片编辑方法"""
        return await self._retry_with_fallback_keys(
            "编辑图片",
            self._edit_image_manually,
            prompt,
            image_path
        )

    async def _generate_image_with_retry(self, prompt, aspect_ratio: str = None, resolution: str = None):
        """带重试逻辑的图片生成方法

        Returns:
            tuple: (image_data, model_text) 或 None
        """
        if aspect_ratio is None:
            aspect_ratio = self.default_aspect_ratio
        if resolution is None:
            resolution = self.default_resolution
        return await self._retry_with_fallback_keys(
            "生成图片",
            self._generate_image_manually,
            prompt,
            aspect_ratio,
            resolution
        )

    async def _edit_image_manually(self, api_key, prompt, image_path):
        """使用 httpx 异步编辑图片"""
        # 使用配置的模型
        model_name = self.model_name

        # 修正 API 地址格式
        base_url = self.api_base_url.strip()
        if not (base_url.startswith("https://") or base_url.startswith("http://")):
            base_url = f"https://{base_url}"
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        endpoint = f"{base_url}/v1beta/models/{model_name}:generateContent"
        logger.info(f"请求地址：{endpoint}")

        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

        # 读取图片并转换为 Base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = (
                base64.b64encode(image_bytes)
                .decode("utf-8")
                .replace("\n", "")
                .replace("\r", "")
            )

        # 构建请求参数（注意：text 和 image 必须在同一个 parts 数组中）
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_base64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 0.8,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 1024,
            },
        }

        # 发送异步请求
        async with httpx.AsyncClient() as client:
            response = await client.post(url=endpoint, json=payload, headers=headers)

        if response.status_code != 200:
            logger.error(
                f"API 编辑请求失败：HTTP {response.status_code}, 响应：{response.text}"
            )
            response.raise_for_status()

        data = response.json()
        # logger.info(f"API 响应数据：{data}") 太 tm 大了

        # 解析图片数据（兼容 Gemini 原生和 OpenAI 风格响应）
        image_data = self._extract_image_from_gemini_response(data)
        if not image_data:
            image_data = self._extract_image_from_openai_response(data)

        if not image_data:
            raise Exception("编辑图片成功，但未获取到图片数据")
        return image_data

    async def _generate_image_manually(self, api_key, prompt, aspect_ratio, resolution):
        """使用 httpx 异步生成图片

        Args:
            api_key: API 密钥
            prompt: 图片描述
            aspect_ratio: 图片比例
            resolution: 分辨率（1k/2k/4k）

        Returns:
            tuple: (image_data, model_text) - 图片数据和模型文本输出
        """
        # 使用配置的模型
        model_name = self.model_name

        base_url = self.api_base_url.strip()
        if not (base_url.startswith("https://") or base_url.startswith("http://")):
            base_url = f"https://{base_url}"
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        endpoint = f"{base_url}/v1beta/models/{model_name}:generateContent"
        logger.info(f"请求地址：{endpoint}，比例：{aspect_ratio}，分辨率：{resolution}")

        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

        # 构建 generationConfig，包含比例和分辨率参数
        generation_config = {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": 0.8,
            "topP": 0.95,
            "topK": 40,
            "maxOutputTokens": 2048,
        }

        # 添加图片比例（Gemini API 使用 aspectRatio）
        if aspect_ratio:
            generation_config["aspectRatio"] = aspect_ratio

        # 添加分辨率配置
        if resolution and resolution in self.SUPPORTED_RESOLUTIONS:
            res_value = self.SUPPORTED_RESOLUTIONS[resolution]
            generation_config["outputImageSize"] = res_value
            logger.info(f"设置分辨率：{resolution} ({res_value}px)")

        # 注意：不需要 role 字段，直接使用 parts 数组
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url=endpoint, json=payload, headers=headers)

        if response.status_code != 200:
            logger.error(
                f"API 生成请求失败：HTTP {response.status_code}, 响应：{response.text}"
            )
            response.raise_for_status()

        data = response.json()
        logger.info(f"API 响应数据：{data}")

        # 解析图片数据和文本（兼容 Gemini 原生和 OpenAI 风格响应）
        image_data = self._extract_image_from_gemini_response(data)
        if not image_data:
            image_data = self._extract_image_from_openai_response(data)

        # 提取模型的文本输出
        model_text = self._extract_text_from_gemini_response(data)

        if not image_data:
            raise Exception("生成图片成功，但未获取到图片数据")

        return image_data, model_text

    def _extract_image_from_gemini_response(self, data):
        """从 Gemini 原生生成式接口响应中提取图片（inline_data 或 inlineData）。

        返回 bytes 或 None。
        """
        try:
            if "candidates" not in data or not data["candidates"]:
                return None
            candidate = data["candidates"][0]
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            for part in parts:
                # 新字段名（文档推荐）
                if "inline_data" in part and part["inline_data"].get("data"):
                    b64 = (
                        part["inline_data"]["data"].replace("\n", "").replace("\r", "")
                    )
                    return base64.b64decode(b64)
                # 兼容旧字段名
                if "inlineData" in part and part["inlineData"].get("data"):
                    b64 = part["inlineData"]["data"].replace("\n", "").replace("\r", "")
                    return base64.b64decode(b64)
        except Exception as e:
            logger.warning(f"解析 Gemini 响应中的图片失败：{e}")
        return None

    def _extract_text_from_gemini_response(self, data):
        """从 Gemini 响应中提取文本内容。

        返回 str 或 None。
        """
        try:
            if "candidates" not in data or not data["candidates"]:
                return None
            candidate = data["candidates"][0]
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            text_parts = []
            for part in parts:
                if "text" in part and part["text"]:
                    text_parts.append(part["text"])
            if text_parts:
                return "\n".join(text_parts)
        except Exception as e:
            logger.warning(f"解析 Gemini 响应中的文本失败：{e}")
        return None

    def _extract_image_from_openai_response(self, data):
        """从 OpenAI 风格响应中提取图片。

        兼容两类结构：
        - 非流式：choices[*].message.content[*].type == "image_url" -> image_url.url == data:image/png;base64,...
        - 流式分片：choices[*].delta.images[*].image_url.url == data:image/png;base64,...

        返回 bytes 或 None。
        """
        try:
            choices = data.get("choices")
            if not choices:
                return None

            def decode_data_url(url: str):
                if not isinstance(url, str):
                    return None
                # 形如 data:image/png;base64,XXXXX 或 data:image/jpeg;base64,XXXXX
                if url.startswith("data:image") and ";base64," in url:
                    b64 = (
                        url.split(";base64,", 1)[1].replace("\n", "").replace("\r", "")
                    )
                    return base64.b64decode(b64)
                return None

            for ch in choices:
                # 流式：delta.images
                delta = ch.get("delta") or {}
                images = delta.get("images") or []
                for img in images:
                    image_url = img.get("image_url") or {}
                    data_url = image_url.get("url")
                    decoded = decode_data_url(data_url)
                    if decoded:
                        return decoded

                # 非流式：message.content 可能是列表，包含 {type: image_url, image_url: {url: data:...}}
                message = ch.get("message") or {}
                content = message.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_url = item.get("image_url") or {}
                            data_url = image_url.get("url")
                            decoded = decode_data_url(data_url)
                            if decoded:
                                return decoded
        except Exception as e:
            logger.warning(f"解析 OpenAI 风格响应中的图片失败：{e}")
        return None

    async def terminate(self):
        """插件卸载时清理临时目录"""
        if os.path.exists(self.save_dir):
            try:
                for file in os.listdir(self.save_dir):
                    os.remove(os.path.join(self.save_dir, file))
                os.rmdir(self.save_dir)
                logger.info(f"插件卸载：已清理临时目录 {self.save_dir}")
            except Exception as e:
                logger.warning(f"清理临时目录失败：{str(e)}")
        logger.info("Gemini 文生图插件已停用")
