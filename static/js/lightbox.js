document.addEventListener("DOMContentLoaded", function () {

    const lightbox = document.getElementById("image-lightbox");
    const lightboxImage = document.getElementById("image-lightbox-image");
    const closeButton = document.querySelector(".image-lightbox-close");

    document.querySelectorAll(".lightbox-trigger").forEach(function (link) {

        link.addEventListener("click", function (event) {
            event.preventDefault();

            const articleImage = link.querySelector("img");

            lightboxImage.src = link.href;
            lightboxImage.alt = articleImage ? articleImage.alt : "";

            lightbox.classList.add("is-open");
            lightbox.setAttribute("aria-hidden", "false");

            document.body.style.overflow = "hidden";
        });

    });

    function closeLightbox() {
        lightbox.classList.remove("is-open");
        lightbox.setAttribute("aria-hidden", "true");

        lightboxImage.src = "";

        document.body.style.overflow = "";
    }

    closeButton.addEventListener("click", closeLightbox);

    lightbox.addEventListener("click", function (event) {
        if (event.target === lightbox) {
            closeLightbox();
        }
    });

    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" &&
            lightbox.classList.contains("is-open")) {
            closeLightbox();
        }
    });

});