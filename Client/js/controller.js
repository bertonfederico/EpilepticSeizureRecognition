// Controllers for slideshow
var slideIndex = 1;
var actualML = ""
showSlides(slideIndex);
showCharts('../Server/outputImg/y_inspection/y_differences.png', 'dataObservation')
showCharts('../Server/outputImg/basic/eeg_standard_deviation.png', 'dataAnalisys')
showChartsML('DecisionTreeClassifier')

function plusSlides(n) {
	showSlides(slideIndex += n);
}

function currentSlide(n) {
  	showSlides(slideIndex = n);
}

function showSlides(n) {
	let i;
	const slides = document.getElementsByClassName("mySlides");
	const dots = document.getElementsByClassName("dot");
	if (n > slides.length) {slideIndex = 1}
	if (n < 1) {slideIndex = slides.length}
	for (i = 0; i < slides.length; i++) {
		slides[i].style.display = "none";
	}
	for (i = 0; i < dots.length; i++) {
		dots[i].className = dots[i].className.replace(" active", "");
	}
	slides[slideIndex-1].style.display = "block";
	dots[slideIndex-1].className += " active";
}


// Controllers for charts
function showCharts(imgPos, imgIcon) {
	document.getElementById(imgIcon).src = imgPos;
}

function showChartsML(imgPos) {
	showCharts("../Server/outputImg/confusion_matrix/" + imgPos + ".png", 'confusion_matrix')
	showCharts("../Server/outputImg/evaluation/" + imgPos + ".png", 'evaluation')
	document.getElementById("csvFile").value = '';
	document.getElementById("prediction").innerHTML = '';
	document.getElementById("probability").innerHTML = '';
	actualML = imgPos
}