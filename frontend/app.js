const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');

let drawing = false;

// ---------------------- drawing logic ----------------------
function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;

  canvas.width = rect.width * ratio;
  canvas.height = rect.height * ratio;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(ratio, ratio);

  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

resizeCanvas();
window.addEventListener('resize', resizeCanvas);
function getCanvasPosition(e) {
    const rect = canvas.getBoundingClientRect();

  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;

  const x = clientX - rect.left;
  const y = clientY - rect.top;

  return { x, y };
}

function startDrawing(e) {
    e.preventDefault();
    drawing = true;
    const position = getCanvasPosition(e);
    ctx.beginPath();
    ctx.moveTo(position.x, position.y);
}

function draw(e) {
    if (!drawing) {
        return
    }
    e.preventDefault();
    const position = getCanvasPosition(e);
    ctx.lineTo(position.x, position.y);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 10;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
}
function stopDrawing(e) {
    e.preventDefault();
    drawing = false;
    ctx.closePath();
}
function clearCanvas() {
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

document.getElementById('clearCanvas').addEventListener('click', clearCanvas)

// ---------------------- sending image to backend ----------------------

document.getElementById("sendData").addEventListener("click", () => {
    canvas.toBlob(async (blob) => {
        const form = new FormData();
        form.append("image", blob, "drawing.png");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: form
            })
            const result = await response.json();

            const div = document.getElementById('prediction-result')
            div.textContent = `Prediction: ${result.prediction}`;
            div.style.fontSize = "50px";
            div.style.textAlign = "center";
            div.style.color = "white";
            document.body.append(div);
            setTimeout(() => {
                document.body.removeChild(div);
            }, 5000);

        }catch(err){
            console.error(err);
        }
        clearCanvas();
    }, "image/png");

})

// ---------------------- Events ----------------------
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseleave', stopDrawing);
canvas.addEventListener('mouseup', stopDrawing);

canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

// ---------------------- UI tweaks  ----------------------

function scrollToSection(id) {
    const section = document.getElementById(id);
    const padding = -30;
    const y = section.getBoundingClientRect().top + padding;
    window.scrollTo({top: y, behavior: 'smooth'});
}

const menuBar = document.querySelector('.menu-bar');
document.addEventListener('mousemove', (e) => {
    if(e.clientY < 50){ // if mouse is within 50 px of the top
        menuBar.classList.add('visible');
    }else {
        menuBar.classList.remove('visible');
    }
});

