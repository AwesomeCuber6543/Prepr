import Navbar from './Navbar';
import { startInterview, endInterview } from './Button'; // Import the Button component
import './App.css'
import Monitor from './Monitor';
import Textbox from './Textbox';


var thingy = false;


function App() {
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
        alert('Go ahead and introduce yourself');
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
        window.location.reload(); 
      })
      .catch(error => {
        console.error('Error:', error);
        alert('Failed to end the interview.');
      });
  }
  

  return (
  <div className="App">
    <Navbar />
    <div className='Monitor'>
      <Monitor />
      <div className='Textbox'>
        <Textbox />
      </div>
    </div>
  
    <div className='Result'>
      <div className='ButtonContainer'>
      <button onClick={startInterview}>Start Interview</button>
      <button onClick={endInterview}>End Interview</button>
      </div>
    </div>
  </div>
);

  
  
  
}

export default App;
