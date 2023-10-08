// Textbox.js

import React, { useState, useEffect } from 'react';

async function getContactPercentage() {
  try {
    const response = await fetch('http://localhost:2516/GetContactPercentage');
    const data = await response.json();
    const eyeContactPercentage = parseFloat(data);
    if (!isNaN(eyeContactPercentage)) {
      return eyeContactPercentage.toFixed(2) + '%';
    } else {
      console.error('Invalid eyeContactPercentage:', data);
      return 'Data not available';
    }
  } catch (error) {
    console.error('Error fetching eye contact data:', error);
    return 'Data not available';
  }
}

async function getKeywordsUsed() {
  try {
    const response = await fetch('http://localhost:2516/getKeyWordUsage');
    const data = await response.json();
    return data.join(', '); // Assuming keywords are returned as an array.
  } catch (error) {
    console.error('Error fetching keywords used data:', error);
    return 'Data not available';
  }
}

async function getNumberOfFillerWords() {
  try {
    const response = await fetch('http://localhost:2516/getFillerWordsUsed');
    const data = await response.json();
    return data; // Assuming the number of filler words is returned as a number.
  } catch (error) {
    console.error('Error fetching filler words data:', error);
    return 'Data not available';
  }
}

function Textbox() {
  const [eyeContactPercentage, setEyeContactPercentage] = useState('Loading...');
  const [keywordsUsed, setKeywordsUsed] = useState('Loading...');
  const [numberOfFillerWords, setNumberOfFillerWords] = useState('Loading...');

  useEffect(() => {
    async function fetchData() {
      const eyeContactResult = await getContactPercentage();
      setEyeContactPercentage(eyeContactResult);

      const keywordsResult = await getKeywordsUsed();
      setKeywordsUsed(keywordsResult);

      const fillerWordsResult = await getNumberOfFillerWords();
      setNumberOfFillerWords(fillerWordsResult);
    }

    fetchData();
  }, []);

  return (
    <div>
      <div>
        <strong>Eye Contact Percentage:</strong> {eyeContactPercentage}
      </div>
      <div>
        <strong>Keywords Used:</strong> {keywordsUsed}
      </div>
      <div>
        <strong>Number of Filler Words Used:</strong> {numberOfFillerWords}
      </div>
    </div>
  );
}

export default Textbox;
