async function fetchUsernameIfNeeded() {
  const existing = localStorage.getItem("generatedUsername");
  const error = localStorage.getItem("usernameError");

  // Already done or failed
  if (existing || error) return;

  const answers = JSON.parse(localStorage.getItem("answers") || "[]");

  if (!answers.length) {
    console.error("No answers found in localStorage");
    localStorage.setItem("usernameError", "true");
    return;
  }

  try {
    const res = await fetch("http://127.0.0.1:8000/generate-username", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answers }),
    });

    if (!res.ok) throw new Error("Backend returned error");
    const data = await res.json();

    localStorage.setItem("generatedUsername", JSON.stringify(data));
    console.log("Fetched + saved username:", data);
  } catch (err) {
    console.error("Failed to fetch username:", err);
    localStorage.setItem("usernameError", "true");
  }
}

// Start fetching when name.html loads
fetchUsernameIfNeeded();

async function submitName() {
  const nameInput = document.querySelector(".name-input");
  const name = nameInput.value.trim();
  const submitButton = document.querySelector(".nav-buttons button");

  if (name === "") {
    alert("Please enter a name!");
    return;
  }

  if (!/^[A-Za-z\s]*$/.test(name)) {
    alert("Please use only letters and spaces!");
    return;
  }

  // Disable the button and show loading text
  submitButton.disabled = true;
  submitButton.textContent = "Generating...";

  // Store the real name
  localStorage.setItem("realName", name);

  // Check what's in localStorage
  console.log("Checking localStorage...");
  console.log("generatedUsername:", localStorage.getItem("generatedUsername"));
  console.log("usernameError:", localStorage.getItem("usernameError"));

  const generatedUsername = localStorage.getItem("generatedUsername");
  const usernameError = localStorage.getItem("usernameError");

  if (generatedUsername) {
    console.log("Username found, redirecting to result.html");
    window.location.href = "result.html";
  } else if (usernameError) {
    alert("Failed to generate username. Please try again.");
    submitButton.disabled = false;
    submitButton.textContent = "SUBMIT";
    return;
  } else {
    let checkCount = 0;
    while (true) {
      checkCount++;
      console.log(`Checking attempt ${checkCount}...`);

      await new Promise((r) => setTimeout(r, 500));

      const result = localStorage.getItem("generatedUsername");
      const error = localStorage.getItem("usernameError");

      console.log("Current result:", result);
      console.log("Current error:", error);

      if (result) {
        console.log("Username ready! Redirecting...");
        window.location.href = "result.html";
        break;
      }

      if (error) {
        submitButton.disabled = false;
        submitButton.textContent = "SUBMIT";
        alert("Failed to generate username. Please try again.");
        break;
      }

      if (checkCount > 240) {
        submitButton.disabled = false;
        submitButton.textContent = "SUBMIT";
        alert(
          "Request is taking too long. Please check your connection and try again."
        );
        break;
      }
    }
  }
}

// Wait for backend-generated username on load
async function waitForUsername() {
  const spinner = document.getElementById("loadingSpinner");
  // Wait indefinitely without timeout
  while (true) {
    const result = localStorage.getItem("generatedUsername");
    if (result) {
      // Username is ready
      return;
    }

    if (localStorage.getItem("usernameError")) {
      alert("Failed to generate username.");
      return;
    }

    // Check every 500ms
    await new Promise((r) => setTimeout(r, 500));
  }
}

// Run on page load
waitForUsername();

// Allow Enter key to submit
document
  .querySelector(".name-input")
  .addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      submitName();
    }
  });

// Real-time validation - remove non-letter characters
document.querySelector(".name-input").addEventListener("input", function (e) {
  e.target.value = e.target.value.replace(/[^A-Za-z\s]/g, "");
});
