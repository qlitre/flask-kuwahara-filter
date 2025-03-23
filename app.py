from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from pathlib import Path
from kuwahara import get_kuwahara_filtered_pic

app = Flask(__name__)
UPLOAD_FOLDER = Path("static")
UPLOAD_FOLDER.mkdir(exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        param = int(request.form.get("parameter", 5))
        if file:
            img = Image.open(file.stream)
            img_array = np.array(img)
            filtered = get_kuwahara_filtered_pic_array(img_array, r=param)
            result_img = Image.fromarray(filtered)

            # 一時保存して表示
            save_path = UPLOAD_FOLDER / "output.png"
            result_img.save(save_path)

            return render_template("index.html", result_image="output.png")

    return render_template("index.html", result_image=None)

def get_kuwahara_filtered_pic_array(image_array, r=5):
    temp_path = UPLOAD_FOLDER / "temp_input.png"
    Image.fromarray(image_array).save(temp_path)
    result_array = get_kuwahara_filtered_pic(str(temp_path), r=r)

    # float32 → uint8 に正しく変換（0〜1の範囲 → 0〜255 にスケーリング）
    if result_array.dtype != np.uint8:
        result_array = (np.clip(result_array, 0.0, 1.0) * 255).astype(np.uint8)

    return result_array


if __name__ == "__main__":
    app.run(debug=True)
