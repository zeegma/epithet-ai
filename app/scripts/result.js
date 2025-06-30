function splitAndCapitalize(usernameStr) {
  return usernameStr
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .split(/[_\s]/g)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

window.onload = async function () {
  const usernameDisplay = document.getElementById("username-display");
  const resultDisplay = document.getElementById("result-display");
  const bodyElement = document.body;

  try {
    const urlParams = new URLSearchParams(window.location.search);
    const urlUsername = urlParams.get("username");
    const urlType = urlParams.get("type");

    let username, personality_type;

    if (urlUsername && urlType) {
      console.log("Live Mode - URL params");
      username = urlUsername;
      personality_type = urlType;
    } else {
      const generatedData = localStorage.getItem("generatedUsername");
      const realName = localStorage.getItem("realName");

      if (generatedData) {
        console.log("Live Mode - localStorage");
        const parsed = JSON.parse(generatedData);
        personality_type = parsed.personality_type;
        const generatedUsernameRaw = parsed.username;

        const prettyUsername = splitAndCapitalize(generatedUsernameRaw);

        if (realName) {
          username =
            Math.random() < 0.5
              ? `${prettyUsername} ${realName}`
              : `${realName} the ${prettyUsername}`;
        } else {
          username = prettyUsername;
        }
      } else {
        // Fallback
        username = "Dramatica Chismikera";
        personality_type = "OA";
      }
    }

    // Load description
    const res = await fetch("data/descriptions.json");
    if (!res.ok) throw new Error("Failed to load description file");
    const descriptions = await res.json();
    const { description, audio: audioPath } = descriptions[personality_type];

    // Apply results to page
    const imagePath = `assets/results/${personality_type.toLowerCase()}.png`;
    bodyElement.style.backgroundImage = `url('${imagePath}')`;
    usernameDisplay.textContent = username;
    const article = /^[aeiou]/i.test(personality_type) ? "an" : "a";
    resultDisplay.textContent = `You are ${article} ${personality_type}! ${description}`;

    // Play looping audio
    const audio = new Audio(audioPath);
    audio.loop = true;
    audio.play().catch((err) => {
      console.warn("Autoplay failed:", err);
    });

  } catch (error) {
    console.error("Result display error:", error);
    bodyElement.style.backgroundColor = "#111";
    usernameDisplay.textContent = "ERROR";
    resultDisplay.textContent = error.message || "Unexpected error.";
  }
};
