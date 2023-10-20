// window.addEventListener('DOMContentLoaded', event => {

//     // Navbar shrink function
//     var navbarShrink = function () {
//         const navbarCollapsible = document.body.querySelector('#mainNav');
//         if (!navbarCollapsible) {
//             return;
//         }
//         if (window.scrollY === 0) {
//             navbarCollapsible.classList.remove('navbar-shrink')
//         } else {
//             navbarCollapsible.classList.add('navbar-shrink')
//         }

//     };

//     // Shrink the navbar 
//     navbarShrink();

//     // Shrink the navbar when page is scrolled
//     document.addEventListener('scroll', navbarShrink);
// })

    //Function to interact with the backend when plotting a graph
    function plotGraph() {
        let dataset = document.querySelector("#dataset").value;
        let graphToPlot = document.querySelector("#graphToPlot").value;
        var layout = {
            autosize: true,
            automargin: true,
        };
        document.getElementById("hm-spinner").style.display = '';//Load button clicked, show spinner
        $.ajax({
            url: '/plot_graph',
            type: 'POST',
            data: {
                'dataset': dataset,
                'graphToPlot': graphToPlot
            },
            beforeSend: function() {
                console.log("Sending POST request")
            },
            complete: function() {
                document.getElementById("hm-spinner").style.display = 'none';//Request is complete so hide spinner
                console.log("AJAX Request Complete.")
            },
            success: function (response) {
                var graph1 = JSON.parse(response);
                Plotly.newPlot('graph1', graph1);
            },
            error: function (error) {
                console.log(error);
            }
        }); 
    }


//Run plotGraph on document load
jQuery(document).ready(plotGraph);

    //Function to interact with the backend when plotting a graph
function calculate() {
    let dataset = document.querySelector("#dataset").value;
    let calculation = document.querySelector("#calculation").value;
    var layout = {
        autosize: true,
        automargin: true,
    };
    $.ajax({
        url: '/calculate',
        type: 'POST',
        data: {
            'dataset': dataset,
            'calculation': calculation
        },
        beforeSend: function() {
            console.log("Sending POST request")
        },
        complete: function() {
            document.getElementById("hm-spinner").style.display = 'none';//Request is complete so hide spinner
            console.log("AJAX Request Complete.")
        },
        success: function (response) {
            console.log(response);
            var result = JSON.parse(response);
            terminalWrite("\n")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("-")
            terminalWrite("\n")
            text_box.scrollTop = text_box.scrollHeight;
            terminalWrite(result);
                //Text area scrollbar anchor **These two lines need to be run each time the text area is updated**
            //var text_box = document.getElementById('terminal');
            text_box.scrollTop = text_box.scrollHeight;
        },
        error: function (error) {
            console.log(error);
        }
    }); 
}


//Text area scrollbar anchor **These two lines need to be run each time the text area is updated**
var text_box = document.getElementById('terminal');
text_box.scrollTop = text_box.scrollHeight;



// Mobile Collapse Nav
$('.primary-menu .navbar-nav .dropdown-toggle[href="#"], .primary-menu .dropdown-toggle[href!="#"] .arrow').on('click', function(e) {
	if ($(window).width() < 991) {
        e.preventDefault();
        var $parentli = $(this).closest('li');
        $parentli.siblings('li').find('.dropdown-menu:visible').slideUp();
        $parentli.find('> .dropdown-menu').stop().slideToggle();
        $parentli.siblings('li').find('a .arrow.show').toggleClass('show');
		$parentli.find('> a .arrow').toggleClass('show');
	}
});
//"sleep()" function
function delay(milliseconds){
    return new Promise(resolve => {
        setTimeout(resolve, milliseconds);
    });
}

// Mobile Menu
$('.navbar-toggler').on('click', function() {
	$(this).toggleClass('show');
});

// Text area scripts
// global vars
let terminal, writeSpeed;

window.addEventListener('load', init);

function init() {
  // default settings
  terminal = document.getElementById("terminal");
  writeSpeed = 45;
  terminalStart();
}

function terminalStart() {
    terminalWrite('-- Welcome to using Sentiment Analyzer --\nResults of calculations and comments/conclusions related to them will be displayed here.\n---------------------------------------------------\n');
  }

function terminalWrite(text) {
    let counter = 0;
    (function writer() {
      terminal.disabled = true;
      if (counter < text.length) {
        let terminalText = (`${(terminal.value).replace("|","")}${text.charAt(counter)}`);
        if (counter !== text.length-1) {
          terminalText = `${terminalText}|`
        }
        terminal.value = terminalText;
        text_box.scrollTop = text_box.scrollHeight;
        counter++;
        setTimeout(writer, writeSpeed);
      } else {
        clearTimeout(writer);
        terminal.disabled = false;
        terminal.blur();
        terminal.focus();
      }
    })();  
  }