// Controllers for slideshow
let slideIndex = 1;
let actualML = 'SupportVectorClassification';
let actualAssessmentFinal = "assessment_";
let actualTrainTest = "train_";
let actualDataAnalysis1 = "../Server/outputImg/pot_min_max/";
let actualDataAnalysis2 = "cat_plot.png";
showSlides(slideIndex);
showChartsDataAnalysis("../Server/outputImg/pot_min_max/", 1);
showCharts('../Server/outputImg/y_inspection/row_number.png', 'dataObservation')
showChartsML('SupportVectorClassification', 1)

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

function showChartsDataAnalysis(imgPos, id) {
	if (id === 1) actualDataAnalysis1 = imgPos
	else if (id === 2) actualDataAnalysis2 = imgPos
	document.getElementById('dataAnalysis').src = actualDataAnalysis1 + actualDataAnalysis2;
}

function showChartsML(imgPos, id) {
	if (id === 1) actualML = imgPos
	else if (id === 2) actualAssessmentFinal = imgPos
	else actualTrainTest = imgPos
	showCharts("../Server/outputImg/confusion_matrix/" + actualAssessmentFinal + actualTrainTest
			+ actualML + ".png", 'confusion_matrix')
	showCharts("../Server/outputImg/evaluation/" + actualAssessmentFinal + actualTrainTest
			+ actualML + ".png", 'evaluation')
	document.getElementById("csvFile").value = '';
	document.getElementById("prediction").innerHTML = '';
}