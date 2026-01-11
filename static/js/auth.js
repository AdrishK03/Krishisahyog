document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');

    // Function to handle login form submission
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Collect form data
            const formData = new FormData(loginForm);
            const data = Object.fromEntries(formData.entries());

            // You can add a loading state for the button here if you want
            const loginBtn = loginForm.querySelector('button[type="submit"]');
            const originalLoginText = loginBtn.textContent;
            loginBtn.textContent = 'Logging in...';
            loginBtn.disabled = true;

            try {
                // Send a POST request to the login route
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded'
                    },
                    body: new URLSearchParams(data).toString()
                });
                
                // If the server redirects, it means login was successful
                if (response.redirected) {
                    window.location.href = response.url;
                } else {
                    // If not redirected, login failed. We'll get the HTML with the error message.
                    const resultHtml = await response.text();
                    
                    // Create a temporary element to parse the HTML and find the error message
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(resultHtml, 'text/html');
                    const errorMessageElement = doc.querySelector('.error-message');
                    const errorText = errorMessageElement ? errorMessageElement.textContent.trim() : 'An unknown error occurred during login. Please try again.';

                    // Display the error message on the page
                    alert(errorText);
                }
            } catch (error) {
                // Handle network or other unexpected errors
                console.error('Error during login:', error);
                alert('An error occurred during login. Please check your network and try again.');
            } finally {
                // Reset button state
                loginBtn.textContent = originalLoginText;
                loginBtn.disabled = false;
            }
        });
    }

    // Function to handle registration form submission
    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const registerBtn = registerForm.querySelector('button');
            const originalRegisterText = registerBtn.textContent;
            registerBtn.textContent = 'Registering...';
            registerBtn.disabled = true;

            const formData = new FormData(registerForm);

            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    body: formData
                });
                
                // If the server redirects, it means registration was successful
                if (response.redirected) {
                    window.location.href = response.url;
                } else {
                    const resultHtml = await response.text();
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(resultHtml, 'text/html');
                    const errorMessageElement = doc.querySelector('.error-message');
                    const errorText = errorMessageElement ? errorMessageElement.textContent.trim() : 'An unknown error occurred during registration. Please try again.';
                    
                    alert(errorText);
                }
            } catch (error) {
                console.error('Registration failed:', error);
                alert('An error occurred during registration. Please try again.');
            } finally {
                registerBtn.disabled = false;
                registerBtn.textContent = originalRegisterText;
            }
        });
    }
});