from flask import Flask, render_template,request
from model import plastic_vision_clasificator
import os
# Configuramos la app y le decimos dónde está la carpeta static
app = Flask(__name__, static_folder="static", static_url_path="/static")

# Ruta principal solamente muestra la página
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            return render_template("index.html", error="No enviaste ninguna imagen.")

        filepath = os.path.join("images_2", file.filename)
        file.save(filepath)

        # --- Aquí llamas tu modelo ---
        nombre_clase, valor_seguridad = plastic_vision_clasificator(filepath)

        valor_seguridad *= 100  # convertir a %
        valor_seguridad = round(valor_seguridad, 2)

        # Enviar al HTML
        return render_template(
            "index.html",
            filename=file.filename,
            pred=nombre_clase,
            seguridad=valor_seguridad
        )

    return render_template("index.html")

# Ejecutamos el servidor
if __name__ == "__main__":
    app.run(debug=True)
