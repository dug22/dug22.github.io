document.addEventListener("click", function (e) {
    if (e.target.classList.contains("copy-btn")) {
        const block = e.target.closest(".code-block");
        const code = block.querySelector("pre code").innerText;

        navigator.clipboard.writeText(code).then(() => {
            e.target.textContent = "Copied!";
            setTimeout(() => e.target.textContent = "Copy", 1500);
        });
    }
});
