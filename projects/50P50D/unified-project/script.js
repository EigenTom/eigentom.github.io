const open_btn = document.querySelector('.open-btn');
const close_btn = document.querySelector('.close-btn');
const nav_side = document.querySelectorAll('.nav');
const nav = document.querySelector('.navigator')

open_btn.addEventListener('click', () => {
    nav.classList.add('hidden');
    nav_side.forEach(nav_element => nav_element.classList.add('visible'));
})

close_btn.addEventListener('click', () => {
    nav.classList.remove('hidden');
    nav_side.forEach(nav_element => nav_element.classList.remove('visible'));
})


const search = document.querySelector('.navigator-search');
const search_btn = document.querySelector('.search-btn');
const search_input = document.querySelector('.search-input');

search_btn.addEventListener('click', () => {
    search.classList.toggle('active')
    search_input.focus()
})