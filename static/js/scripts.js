window.addEventListener('DOMContentLoaded', event => {

    // Navbar shrink function
    var navbarShrink = function () {
        const navbarCollapsible = document.body.querySelector('#mainNav');
        if (!navbarCollapsible) {
            return;
        }
        if (window.scrollY === 0) {
            navbarCollapsible.classList.remove('navbar-shrink')
        } else {
            navbarCollapsible.classList.add('navbar-shrink')
        }

    };

    // Shrink the navbar 
    navbarShrink();

    // Shrink the navbar when page is scrolled
    document.addEventListener('scroll', navbarShrink);
})

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
            url: '/dashboard',
            type: 'POST',
            data: {
                'token': token,
                'orderbook': orderbook,
                'tick_interval': tick_interval
            },
            beforeSend: function() {
                console.log("Sending POST request")
            },
            complete: function() {
                document.getElementById("hm-spinner").style.display = 'none';//Request is complete so hide spinner
                console.log("AJAX Request Complete.")
            },
            success: function (response) {
                //console.log(response);
                //$("#chart_heatmap").fadeOut(100).fadeOut(100);
                var graph1 = JSON.parse(response);
                Plotly.newPlot('chart_heatmap', graph1);
            },
            error: function (error) {
                console.log(error);
            }
        }); 
    }


//Run plotGraph on document load
jQuery(document).ready(plotGraph);

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


// Mobile Menu
$('.navbar-toggler').on('click', function() {
	$(this).toggleClass('show');
});
