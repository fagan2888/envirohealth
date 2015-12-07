<?php
  if($_POST['calculate'] == "Submit")
  {
    $errorMessage = "";

    if(empty($_POST['age']))
    {
      $errorMessage .= "<li>Please enter your age"</li>";"
    }
    if(empty($_POST['radtn']))
    {
      $errorMessage .= "<li>Please select a value for radiation treatment</li>";
    }
    if(empty($_POST['race']))
    {
      $errorMessage .= "<li>Please enter your race</li>";
    }
    if(empty($_POST['laterality']))
    {
      $errorMessage .= "<li>Please select laterality status</li>";
    }
    if(empty($_POST['tumorbehavior']))
    {
      $errorMessage .= "<li>Please select tumor behavior</li>";
    }
    if(empty($_POST['tumorstage']))
    {
      $errorMessage .= "<li>Please select a tumor stage</li>";
    }
    if(empty($_POST['numprims']))
    {
      $errorMessage .= "<li>Please select number of primaries</li>";
    }
    if(empty($_POST['erstatus']))
    {
      $errorMessage .= "<li> Please select your ER status</li>";
    }
    if(empty($_POST['prstatus']))
    {
      $errorMessage .= "<li> Please select your PR status</li>";
    }
    $varAge = $_POST['age'];
    $varRadtn = $_POST['radtn'];
    $varRace = $_POST['race'];
    $varLaterality = $_POST['laterality'];
    $varTumorbehavior = $_POST['tumorbehavior'];
    $varTumorstage = $_POST['tumorstage'];
    $varNumberprims = $_POST['numberprims'];
    $varErstatus = $_POST['erstatus'];
    $varPrstatus = $_POST['prstatus'];
    if(!empty($errorMessage))
    {
      echo("<p>Calculation error:<p>\n");
      echo("<ul>" . $errorMessage . "</ul>\n");
    }
  }
  ?>
