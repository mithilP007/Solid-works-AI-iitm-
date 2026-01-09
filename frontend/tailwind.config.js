/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                background: "hsl(240 10% 3.9%)",
                sidebar: {
                    DEFAULT: "hsl(240 5.9% 10%)",
                    accent: "hsl(240 4.8% 15.1%)",
                    border: "hsl(240 3.7% 15.9%)",
                },
                primary: {
                    DEFAULT: "hsl(187 100% 50%)",
                },
                accent: {
                    DEFAULT: "hsl(155 100% 50%)",
                }
            }
        },
    },
    plugins: [],
}
