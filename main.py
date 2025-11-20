import base64
import os
import uuid
import httpx

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, register


@register("AstrBot_Plugins_Canvas", "stevessr", "一个普通的使用 gemini-2.5-flash-image 进行图片内容创作的插件", "v1.1")
class GeminiImageGenerator(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        """Gemini 图片生成与编辑插件初始化"""
        super().__init__(context)
        self.config = config

        logger.info(f"插件配置加载成功：{self.config}")

        # 读取多密钥配置
        self.api_keys = self.config.get("gemini_api_keys", [])
        self.current_key_index = 0

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

    @filter.command("gemini_image", alias={"文生图"})
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        """根据文本描述生成图片"""
        if not self.api_keys:
            yield event.plain_result("错误：未配置任何 Gemini API 密钥")
            return

        if not prompt.strip():
            yield event.plain_result(
                "请输入图片描述（示例：/gemini_image 一只戴帽子的猫在月球上）"
            )
            return

        save_path = None

        try:
            yield event.plain_result("正在生成图片，请稍等...")
            image_data = await self._generate_image_with_retry(prompt)

            if not image_data:
                logger.error("生成失败：所有 API 密钥均尝试完毕")
                yield event.plain_result("生成失败：所有 API 密钥均尝试失败")
                return

            # 保存图片
            file_name = f"{uuid.uuid4()}.png"
            save_path = os.path.join(self.save_dir, file_name)

            with open(save_path, "wb") as f:
                f.write(image_data)

            logger.info(f"图片已保存至：{save_path}")

            # 发送图片
            yield event.chain_result([Image.fromFileSystem(save_path)])
            logger.info(f"图片发送成功，提示词：{prompt}")

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
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str):
        """根据文本描述生成图片，当你需要生成图片时请使用此工具。

        Args:
            prompt(string): 图片描述文本（例如：画只猫）
        """
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

    async def _generate_image_with_retry(self, prompt):
        """带重试逻辑的图片生成方法"""
        return await self._retry_with_fallback_keys(
            "生成图片",
            self._generate_image_manually,
            prompt
        )

    async def _edit_image_manually(self, prompt, image_path, api_key):
        """使用 httpx 异步编辑图片"""
        model_name = "gemini-2.5-flash-image"

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

    async def _generate_image_manually(self, prompt, api_key):
        """使用 httpx 异步生成图片"""
        model_name = "gemini-2.5-flash-image"

        base_url = self.api_base_url.strip()
        if not (base_url.startswith("https://") or base_url.startswith("http://")):
            base_url = f"https://{base_url}"
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        endpoint = f"{base_url}/v1beta/models/{model_name}:generateContent"
        logger.info(f"请求地址：{endpoint}")

        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

        # 注意：不需要 role 字段，直接使用 parts 数组
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 0.8,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 1024,
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url=endpoint, json=payload, headers=headers)

        if response.status_code != 200:
            logger.error(
                f"API 生成请求失败：HTTP {response.status_code}, 响应：{response.text}"
            )
            response.raise_for_status()

        data = response.json()
        logger.info(f"API 响应数据：{data}")

        # 解析图片数据（兼容 Gemini 原生和 OpenAI 风格响应）
        image_data = self._extract_image_from_gemini_response(data)
        if not image_data:
            image_data = self._extract_image_from_openai_response(data)

        if not image_data:
            raise Exception("生成图片成功，但未获取到图片数据")
        return image_data

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
