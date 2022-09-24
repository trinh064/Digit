// wait for the content of the window element
// to load, then performs the operations.
// This is considered best practice.
window.addEventListener('load', ()=>{

    //resize(); // Resizes the canvas once the window loads
    document.addEventListener('mousedown', startPainting);
    document.addEventListener('mouseup', stopPainting);
    document.addEventListener('mousemove', sketch);

});

const canvas = document.querySelector('#myCanvas');

// Context for the canvas for 2 dimensional operations
const ctx = canvas.getContext('2d');
//ctx.scale(28/ 200, 28/200);

// Resizes the canvas to the available size of the window.

// Stores the initial position of the cursor
let coord = {x:0 , y:0};

// This is the flag that we are going to use to
// trigger drawing
let paint = false;

// Updates the coordianates of the cursor when
// an event e is triggered to the coordinates where
// the said event is triggered.
function getPosition(event){
  coord.x = event.clientX - canvas.offsetLeft;
  coord.y = event.clientY - canvas.offsetTop;
}

// The following functions toggle the flag to start
// and stop drawing
function startPainting(event){
  paint = true;
  getPosition(event);
}
function stopPainting(){
  paint = false;

}

function sketch(event){
  if (!paint) return;
  ctx.beginPath();

  ctx.lineWidth = 5;

  // Sets the end of the lines drawn
  // to a round shape.
  ctx.lineCap = 'round';

  ctx.strokeStyle = 'red';

  // The cursor to start drawing
  // moves to this coordinate
  ctx.moveTo(coord.x, coord.y);

  // The position of the cursor
  // gets updated as we move the
  // mouse around.
  getPosition(event);

  // A line is traced from start
  // coordinate to this coordinate
  ctx.lineTo(coord.x , coord.y);

  // Draws the line.
  ctx.stroke();
}

// Setting up tfjs with the model we downloaded
tf.loadLayersModel('model/model.json')
	.then(function (model) {
		window.model = model;
	});


function cleancanvas () {
		console.log("clear");
		ctx.clearRect(0, 0, 500, 500);
	}

// Predict function
function predict () {
	//var img = new Image();
  //ctx.drawImage(img,0,0,28,28);
	console.log("predict");
  data = ctx.getImageData(0,0,224,224).data;
	var input = [];
        for (var i = 0; i < data.length; i += 4) {
            input.push(data[i]);
					//	if(data[i] || data[i+1] || data[i+2] || data[i+3] ){console.log(i);}
        }
  console.log(input);
	if (window.model) {
		window.model.predict([tf.tensor(input)
			.reshape([1,224,224,1])])
			.array().then(function (scores) {

				scores = scores[0];
				predicted = scores
					.indexOf(Math.max(...scores));
				$('#number').html(predicted);
			});
	} else {
		// The model takes a bit to load,
		// if we are too fast, wait
		setTimeout(function () { predict(input) }, 50);
	}
}
