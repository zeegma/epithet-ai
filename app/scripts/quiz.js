let questions = [];
const answerList = [];
let currentIndex = 0;

const buttonImages = [
  "assets/buttons/button_a.png",
  "assets/buttons/button_b.png",
  "assets/buttons/button_c.png",
  "assets/buttons/button_d.png",
];

function preloadBackgroundImages() {
  questions.forEach((q) => {
    const img = new Image();
    img.src = q.bg;
  });
}

async function loadQuestions() {
  try {
    const res = await fetch("data/questions.json");
    const data = await res.json();
    questions = data.questions;
    preloadBackgroundImages();
    renderQuestion(currentIndex, true);
  } catch (err) {
    alert("Failed to load questions.json");
    console.error(err);
  }
}

function renderQuestion(index, skipAnimation = false) {
  const container = document.getElementById("quiz-container");

  if (!skipAnimation) {
    container.classList.add("fade-out");
    setTimeout(() => {
      container.classList.remove("fade-out");
      container.classList.add("fade-in");
      actuallyRender(index);
      setTimeout(() => container.classList.remove("fade-in"), 300);
    }, 300);
  } else {
    actuallyRender(index);
  }
}

function actuallyRender(index) {
  const container = document.getElementById("quiz-container");
  container.innerHTML = "";

  const q = questions[index];
  const options = [q.A, q.B, q.C, q.D];

  const bg = new Image();
  bg.src = q.bg;
  bg.onload = () => {
    document.body.style.backgroundImage = `url(${q.bg})`;
  };

  options.forEach((text, i) => {
    const btn = document.createElement("button");
    btn.className = "image-button";
    if (answerList[index] === i + 1) {
      btn.classList.add("selected");
    }

    btn.onclick = () => {
      answerList[index] = answerList[index] === i + 1 ? null : i + 1;
      renderQuestion(index);
    };

    const img = document.createElement("img");
    img.src = buttonImages[i];

    const span = document.createElement("span");
    span.className = "button-text";
    span.textContent = `${String.fromCharCode(65 + i)}) ${text}`;

    btn.appendChild(img);
    btn.appendChild(span);
    container.appendChild(btn);
  });

  document.getElementById("prevBtn").style.display =
    index === 0 ? "none" : "inline-block";

  const nextBtn = document.getElementById("nextBtn");
  nextBtn.textContent = index === questions.length - 1 ? "SUBMIT" : "NEXT";

  // Disable next if not answered
  nextBtn.disabled = !answerList[index];
  nextBtn.style.opacity = answerList[index] ? "1" : "0.5";
  nextBtn.style.pointerEvents = answerList[index] ? "auto" : "none";
}

function nextQuestion() {
  if (currentIndex < questions.length - 1) {
    currentIndex++;
    renderQuestion(currentIndex);
  } else {
    // Save answers to localStorage
    localStorage.setItem("answers", JSON.stringify(answerList));
    localStorage.removeItem("generatedUsername");
    localStorage.removeItem("usernameError");

    // Immediately go to name.html
    window.location.href = "name.html";
  }
}

function prevQuestion() {
  if (currentIndex > 0) {
    currentIndex--;
    renderQuestion(currentIndex);
  }
}

loadQuestions();
