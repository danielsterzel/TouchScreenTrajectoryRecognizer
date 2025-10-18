function scrollToSection(id) {
    const section = document.getElementById(id);
    const padding = -30;
    const y =  section.getBoundingClientRect().top + padding;
    window.scrollTo({ top: y, behavior: 'smooth' });
}