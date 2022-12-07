function darkMode() {
    var element = document.body;
    var help_button = document.getElementById("help_theme");
    var choose_file = document.getElementById("choose_file_theme");
    var upload_button = document.getElementById("upload_theme");
    var header = document.getElementById("header");
    // var content = document.getElementById("body");
    element.className = "dark_mode";
    help_button.className = "help_dark";
    choose_file.className = "upload_dark";
    upload_button.className = "upload_dark";
}

function lightMode() {
    var element = document.body;
    var help_button = document.getElementById("help_theme");
    var choose_file = document.getElementById("choose_file_theme");
    var upload_button = document.getElementById("upload_theme");
    // var content = document.getElementById("body");
    element.className = "light_mode";
    help_button.className = "help_light";
    choose_file.className = "upload_light";
    upload_button.className = "upload_light";
}
