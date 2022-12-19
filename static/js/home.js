// Get the root element
var r = document.querySelector(":root");
var preview = document.querySelector("img");
var preview_text = document.querySelector("h2");

function lightMode() {
    // Set the value of variable --blue to the value lightblue - From: https://www.w3schools.com/css/css3_variables_javascript.asp
    r.style.setProperty("--background-colour", "#F2F2F2");
    r.style.setProperty("--background-text", "#0D0D0D");
    r.style.setProperty("--title-colour", "#7A8FA3");
    r.style.setProperty("--header-colour", "#031626");
    r.style.setProperty("--button-colour", "#7A8FA3");
    r.style.setProperty("--button-border", "#031626");
    r.style.setProperty("--button-text", "var(--background-text)");
}

function darkMode() {
    r.style.setProperty("--background-colour", "#0D0D0D");
    r.style.setProperty("--background-text", "#F2F2F2");
    r.style.setProperty("--title-colour", "#010326");
    r.style.setProperty("--header-colour", "#566573");
    r.style.setProperty("--button-colour", "#031626");
    r.style.setProperty("--button-border", "#F2F2F2");
    r.style.setProperty("--button-text", "var(--background-text)");
}

document.getElementById("choose_file_theme").onchange = function () {   // https://stackoverflow.com/questions/14069421/show-an-image-preview-before-upload/14069481#14069481
    var src = URL.createObjectURL(this.files[0]);
    preview_element = document.getElementById("preview_image");
    preview_element.src = src;
    preview.style.setProperty("width", "30%");
    preview.style.setProperty("height", "auto");
    preview.style.setProperty("border", "5px solid var(--button-border)");
    preview.style.setProperty("margin", "5px 35% 0 35%");
    preview_text.style.setProperty("display", "block");
}