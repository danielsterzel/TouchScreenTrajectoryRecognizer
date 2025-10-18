const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');

let drawing = false;
let points = [];

function getCanvasPosition(e) {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;
    return {x, y}
}

function startDrawing(e) {
    e.preventDefault();
    drawing = true;
    points = [];
    const position = getCanvasPosition(e);
    points.push({x: position.x, y: position.y, t: Date.now()});
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
    ctx.lineWidth = 2;
    ctx.stroke();
    points.push({x: position.x, y: position.y, t: Date.now()});
}
function stopDrawing(e) {
    e.preventDefault();
    drawing = false;
    ctx.closePath();
    console.log("Zarejestrowane punkty: ", points);
}

document.getElementById('clearCanvas').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    points = [];
});

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