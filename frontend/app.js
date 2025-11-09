const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');

let drawing = false;
let points = [];

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;

  canvas.width = rect.width * ratio;
  canvas.height = rect.height * ratio;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(ratio, ratio);

  clearCanvas();
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
    points = [];
    const position = getCanvasPosition(e);
    ctx.beginPath();
    ctx.moveTo(position.x, position.y);
    points.push({x: position.x, y: position.y, t: Date.now()});
}

function draw(e) {
    if (!drawing) {
        return
    }
    e.preventDefault();
    const position = getCanvasPosition(e);
    ctx.lineTo(position.x, position.y);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
    points.push({x: position.x, y: position.y, t: Date.now()});
}
function stopDrawing(e) {
    e.preventDefault();
    drawing = false;
    ctx.closePath();
    console.log("Registered points: ", points);
}
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    points = [];
}
document.getElementById('clearCanvas').addEventListener('click', clearCanvas);

function returnJson(points){
    return points.map(point => ({x: point.x, y:point.y, t: point.t}))
}

document.getElementById('sendData').addEventListener('click', () => {
    const data = returnJson(points);

    fetch("/submit-points", {
        method: "POST",
        headers: {"Content-Type" : "application/json"},
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(result => {
            console.log("Server response: ", result)
            const p = document.createElement('p');
            p.textContent = "Data sent successfully!";
            p.style.color = "lime";
            p.style.fontSize = "30px";
            p.style.zIndex = "9999";
            p.style.top = "50%";
            p.style.left = "50%";
            document.body.appendChild(p);
            setTimeout(() => {
                document.body.removeChild(p);
            }, 4000)
        })
        .catch(err => console.error("Error sending points: ", err));
    clearCanvas();
})

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseleave', stopDrawing);
canvas.addEventListener('mouseup', stopDrawing);

canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

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

