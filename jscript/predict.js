/* Notes:

In order to get a prediction image that has the same high resolution as the
input image the following is important:

1. The code (file reader) reads the input image from the img tag. Do not resize the 
input image in any way using the img tag i.e. don't set height or width. 
Let the image be loaded in its full actual size. This big image can be hidden.

2. Add the predicted mask as a 4th channel to the original big image not
to any image that has been resized. The 
predicted mask (alpha image) will need to be resized to the full size of the original big
image before it gets added as a 4th channel.
After adding this predicted mask as a 4th channel, the resulting 4 channel 
image can then be resized before displaying on the page.



To make the segmented areas visible only:

- Overlay 3 images: original image, color image with opacity, the segmented image.
- Each of these images is on a canvas.
- We use the CSS z-index to control the order of the images.
- Because all images are on a canvas, we need to read and convert each image to a tensor.
- This tensor is then written to the canvas.
- We enclose the 3 canvas elements (absolute position) inside a div elemnet that has relative position. The canvases are then positioned realative to the div element - they are fenced
	inside the div element. This div is also responsive, canvas elements with position absolute don't behave responsively.

*/




// =================
// Define Functions
// =================




// == Simulate a click on a hidden button. == //
function simulateClick(tabID) {
	
	document.getElementById(tabID).click();
}



// == Starts predicting immediately when an image is submitted. == //
function predictOnLoad() {
	
	// Simulate a click on the predict button
	setTimeout(simulateClick.bind(null,'predict-button'), 500);
};



// Hide the seg image loading spinner
$('.spinner').hide();


// == Detects when a new image has been submitted. == //
$("#image-selector").change(function () {
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		
		// display the user submitted image on the page by changing the src attribute.
		$("#selected-image").attr("src", dataURL);
		$("#displayed-image").attr("src", dataURL);
		// empty the prediction list.
		$("#prediction-list").empty();
		
		// When a new image is submitted this code clears 
		// the previous segmentation pred
		// by covering the canvas with a white image.
		var canvas = document.getElementById("myCanvas2");
  		var ctx = canvas.getContext("2d");
  		var img = document.getElementById("color-image");
  		ctx.drawImage(img, 0, 0);
		
		// Show the seg image loading spinner
		$('.spinner').show();
		
		
	}
	
		
		let file = $("#image-selector").prop('files')[0];
		reader.readAsDataURL(file);
		
		
		// Simulate a click on the predict button
		// This introduces a 0.5 second delay before the click.
		// Without this long delay the model loads but may not automatically
		// predict.
		setTimeout(simulateClick.bind(null,'predict-button'), 500);

});



// == Loads the model and makes a prediction on the default image. == //
let model;
(async function () {
	
	model = await tf.loadModel('http://brain.test.woza.work/model_3/model.json');
	$("#selected-image").attr("src", "http://brain.test.woza.work/assets/ich.jpg");
	
	
	// Hide the model loading spinner
	$('.progress-bar').hide();
	
	// Show the seg image loading spinner
	$('.spinner').show();
	
	// Simulate a click on the predict button
	predictOnLoad();
	
	
})();





// == Responds to a click on the hidden predict button == //
$("#predict-button").click(async function () {
	
	
	// User submitted image
	let image = $('#selected-image').get(0);
	
	// Get the color image
	let blank_image = $('#color-image').get(0);
	
	

	
	
	// ========================================================== //
	// Pre-process the image using Tensorflow.js api. This is not standard Tensorflow.
	// This is tensorflow.js code designed to be used with javascript.
	// https://js.tensorflow.org/api/0.6.1/#slice
	// ========================================================== //
	
	// This is how we print info on tensors to the console:
	// verbose can be left out e.g. tensor.print()
	const verbose = true;
	//tensor.print(verbose);
	
	
	
	// Keep a copy of the original image for later.
	var orig_image = tf.fromPixels(image);
	
	// This is the color image
	var color_image = tf.fromPixels(blank_image);
	
	
	orig_image.print(verbose);
	
	
	
	// Pre-process the image
	let tensor = tf.fromPixels(image)
	.resizeNearestNeighbor([256,256]) // change the image size here
	.toFloat()
	.div(tf.scalar(255.0))
	.expandDims();
	
	
	console.log('Input Image shape: ', tensor.shape);
	
		
		
	// Pass the tensor to the model and call predict on it.
	// Predict returns a tensor.
	// data() loads the values of the output tensor and returns
	// a promise of a typed array when the computation is complete.
	// Notice the await and async keywords are used together.
	
	// ========================================================== //
	// Note that after predict is completed we are working with
	// a Typed Array and not a tensor.
	// ========================================================== //
		
	// make a prediction
	// model.predict returns a 'Typed Array'. This is not a tensor or js array.
	let predictions = await model.predict(tensor).data();
		
		
		
	// convert typed array to a javascript array
	var preds = Array.from(predictions); // JS Code with js array
		
	
	// threshold the predictions
	var i;
	var num;
	for (i = 0; i < preds.length; i++) { 
		
		num = preds[i];
		
		if (num < 0.7) {   // <-- Set the threshold here
			preds[i] = 0;
			
		} else {
			preds[i] = 255;
			
		}
		
	}
		
		
	// convert js array to a tensor
	pred_tensor = tf.tensor1d(preds, 'int32');
	
	
	// reshape the pred tensor
	pred_tensor = pred_tensor.reshape([256,256,1]);
	
	// resize pred_tensor
	pred_tensor = pred_tensor.resizeNearestNeighbor([orig_image.shape[0], orig_image.shape[1]]);
	
	// reshape the input image tensor
	//input_img_tensor = tensor.reshape([128,128,3]);
	
	// append the tensor to the input image to create a 4th alpha channel --> shape [128,128,4]
	rgba_tensor = tf.concat([orig_image, pred_tensor], axis=-1);
	
	
	// resize all images. rgb_tensor is the segmented image.
	rgba_tensor = rgba_tensor.resizeNearestNeighbor([250, 250]);
	orig_image = orig_image.resizeNearestNeighbor([250, 250]);
	color_image = color_image.resizeNearestNeighbor([250, 250]);
	
	
	
	
	// Convert the tensor to an image 
	
	
	// Method 1: Canvas - Display pred image using tf.toPixels
	//=====================================================
	
	var canvas2 = document.getElementById("myCanvas2");
	var canvas3 = document.getElementById("myCanvas3");
	var canvas4 = document.getElementById("myCanvas4");
	
	// Write all images to 3 seperate canvas elements
	// They will be displayed one on top of another.
	tf.toPixels(rgba_tensor, canvas2);
	tf.toPixels(orig_image, canvas3);
	tf.toPixels(color_image, canvas4);
	

	// Hide the model loading spinner
	$('.spinner').hide();
	
	// Method 2: Canvas - Display pred image using custom code shared on Github
	// https://github.com/tensorflow/tfjs/issues/865
	// This needs a 4 channel image to work.
	//======================================================
	/*
	var canvas = document.getElementById("myCanvas");
	const bytes = await rgba_tensor.data();
	const pixelData = new Uint8ClampedArray(bytes);
	if (canvas !== null){
	    const imageData = new ImageData(pixelData, 250, 250);
	    const ctx = canvas.getContext('2d');
	    ctx.putImageData(imageData, 0, 0);
	}
   */


});









