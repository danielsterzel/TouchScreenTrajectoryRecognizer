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
    const json = JSON.stringify(data, null, 2);

    // creates a blob object - Binary Large Object in memory.
    const blob = new Blob([json], {type: 'application/json'});

    const url = URL.createObjectURL(blob); // in order for the file to be downloadable
    // we create a temporary URL that points to the blob object in memory
    // returns a temporary URL string
    // the url exists only in the current browser tab and until
    // you revoke it with URL.revokeObjectURL(url)
    const ts = new Date().toISOString().replace(/[:.]/g,  '-').replace(/T/g," At ");
    const a = document.createElement('a');
    const p = document.createElement('p');
    p.textContent = "Data was sent successfully!";
    p.style.zIndex = "9999";
    p.style.color = "lime";
    p.style.fontSize = "30px";
    p.style.top = "50%";
    p.style.left = "50%";
    p.style.position = "absolute";
    document.body.appendChild(p);
    setTimeout(() => {
        document.body.removeChild(p);
    }, 4000)
    a.href = url;
    a.download = `points-${ts}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

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

