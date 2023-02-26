// Get the root element
var r = document.querySelector(":root");

function lightMode() {
    // Set the value of variable --blue to the value lightblue - From: https://www.w3schools.com/css/css3_variables_javascript.asp
    r.style.setProperty("--background-colour", "#F2F2F2");
    r.style.setProperty("--background-text", "#0D0D0D");
    r.style.setProperty("--title-colour", "#7A8FA3");
    r.style.setProperty("--header-colour", "#031626");
    r.style.setProperty("--button-colour", "#7A8FA3");
    r.style.setProperty("--button-border", "#031626");
    r.style.setProperty("--button-text", "var(--background-text)");

    localStorage.setItem("theme", "light");
}

function darkMode() {
    r.style.setProperty("--background-colour", "#0D0D0D");
    r.style.setProperty("--background-text", "#F2F2F2");
    r.style.setProperty("--title-colour", "#010326");
    r.style.setProperty("--header-colour", "#566573");
    r.style.setProperty("--button-colour", "#031626");
    r.style.setProperty("--button-border", "#F2F2F2");
    r.style.setProperty("--button-text", "var(--background-text)");

    localStorage.setItem("theme", "dark");
}

//* From: https://css-tricks.com/a-complete-guide-to-dark-mode-on-the-web
//* Either have it automatically set based on the user preference
// const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
// if (prefersDarkScheme.matches) {
//     darkMode();
// } else {
//     lightMode();
// }

//* Or set it based on what they decide
const currentTheme = localStorage.getItem("theme");
// If the current theme in localStorage is "dark"
if (currentTheme == "dark") {
    darkMode();
} else {
    lightMode();
}
