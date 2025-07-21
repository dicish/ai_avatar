import os
import torch
from gradio_client import Client
import streamlit as st
import time
import cv2
from PIL import Image


text = st.text_input("Введите сообщение для озвучки:")


def validate_image(image_path):
    """Проверяем и исправляем изображение перед обработкой"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")

        # Конвертируем в RGB, если нужно
        if len(img.shape) == 2:  # Градации серого
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # С альфа-каналом
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Сохраняем исправленное изображение
        fixed_path = os.path.join(os.path.dirname(image_path), "fixed_" + os.path.basename(image_path))
        Image.fromarray(img).save(fixed_path)
        return fixed_path
    except Exception as e:
        raise RuntimeError(f"Ошибка обработки изображения: {str(e)}")


def text_to_speech(text, output_path='output.wav'):
    language = 'ru'
    model_id = 'v3_1_ru'
    speaker = 'baya'

    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language=language,
        speaker=model_id
    )

    model.save_wav(
        text=text,
        speaker=speaker,
        sample_rate=48000,
        audio_path=output_path
    )
    return os.path.abspath(output_path)


def animate_with_sadtalker(image_path, audio_path, server_url="http://127.0.0.1:7860/"):
    client = Client(server_url)
    time.sleep(10)  # Ждём инициализации сервера

    try:
        # Параметры SadTalker
        result = client.predict(
            image_path,
            audio_path,
            "full",
            False,
            True,
            1,
            256,
            0,
            fn_index=0
        )
        return result
    except:
        print("⏳ 2 минуты и ваше видео появится в results")
        return None


if text:

    try:

        st.info("🎙 Создание аудио...")

        audio_file = text_to_speech(text)

        st.info(f"✅ Аудио создано: {audio_file}")

        image_file = os.path.abspath('test.jpg')

        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Изображение не найдено: {image_file}")

        st.info("🖼 Проверка изображения...")

        fixed_image = validate_image(image_file)

        st.info(f"✅ Изображение обработано: {fixed_image}")

        st.info("🎞 Создание анимации...")

        result_video = animate_with_sadtalker(fixed_image, audio_file)

        if result_video:

            st.info(f"🎬 Готово! Видео сохранено: {result_video}")

        else:

            st.info("ℹ️ Проверьте папку results — видео появится там автоматически.")


    except Exception as e:

        st.info(f"❌ Ошибка: {str(e)}")
