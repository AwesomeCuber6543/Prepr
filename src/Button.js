// button.js

import './Button.css'

// Function to start the interview
function startInterview() {
    // Make an API call using the fetch API
    fetch('http://localhost:2516/StartInterview')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Handle the API response data here
            console.log(data);
            alert('Interview started successfully!');
        })
        .catch(error => {
            // Handle any errors that occurred during the fetch
            console.error('Error:', error);
            alert('Failed to start the interview.');
        });
}

function endInterview() {
    fetch('http://localhost:2516/EndInterview')
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        console.log(data);
        alert('Interview ended successfully!');
      })
      .catch(error => {
        console.error('Error:', error);
        alert('Failed to end the interview.');
      });
  }
  

// Export the startInterview function
export { startInterview, endInterview };