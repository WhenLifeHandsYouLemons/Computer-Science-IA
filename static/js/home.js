var preview = document.querySelector("img");
var preview_text = document.querySelector("h2");

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
