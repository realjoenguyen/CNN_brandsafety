var feedback_link = "https://lum-staging.knorex.com/content-extraction";
// var feedback_link = "http://127.0.0.1:19007/content-extraction";
var coundown_time = 240;
var noti = 'If wrong, please choose the most suitable category:';

if (!(window.bootbox && window.jQuery)) {
  var script_srcs = [
    'https://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js',
    'https://lum-staging.knorex.com/static/bookmarklet/js/bootstrap.min.js',
    'https://lum-staging.knorex.com/static/bookmarklet/js/bootbox.min.js'
//    'http://127.0.0.1:8000/git/bookmarklet/js/bootstrap.js',
//    'http://127.0.0.1:8000/git/bookmarklet/js/bootbox.js'
  ];

  var css = [
//    'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css'
    'https://lum-staging.knorex.com/static/bookmarklet/css/bootstrap_all_min_v1.css'
//    'http://127.0.0.1:8000/git/bookmarklet/css/bootstrap.21012015.css'
//    'http://127.0.0.1:8000/git/bookmarklet/test/css/bootstrap.22012015.css'
    ]


  for (var i = 0; i < css.length; i++) {
    var link = document.createElement('link');
    link.href = css[i];
    link.rel = 'stylesheet';
    document.head.appendChild(link);
  }


  var modaling = document.createElement("div");
  modaling.className = "modaling";
  document.getElementsByTagName('body')[0].appendChild(modaling);
  document.getElementsByTagName('body')[0].className = "loading";

  for (var i = 0; i < script_srcs.length; i++) {
    var xhrObj = new XMLHttpRequest();
    // open and send a synchronous request
    xhrObj.open('GET', script_srcs[i], false);
    xhrObj.send('');

    var script = document.createElement('script');
    script.type = 'text/javascript';
    script.text = xhrObj.responseText;
    document.getElementsByTagName('head')[0].appendChild(script);
  }

  jQuery(function ($) {
    $(document).ready(function() {
      sendRequest();
    })
  });
}
else {
  sendRequest();
}

function sendRequest() {
  var url = location.href;
  jQuery(function ($) {
    $(document).ready(function() {
        if (!$(".modaling").length) {
            $("body").append("<div class='modaling'></div>");
        }
        if (!$(".loading").length) {
            $("body").addClass("loading");
        }

        // Add waiting gif image
        $.ajax({
            url: feedback_link + "/rest/get_category",
            jsonp: 'callback',
            dataType: 'jsonp',
            data: {'url': url},
            success: function(data) {
              // Remove waiting gif image
              $("body").removeClass("loading");
              $('.modaling').fadeOut(500);

              if (data) {
                  var categories = JSON.parse(data["categories"]);
                  var id = data["_id"];

                  if (data["error"]) {
                    bootbox.alert("Oops! It seems the program can't extract content from this site.");
                        window.setTimeout(function() {
                          bootbox.hideAll();
                       }, 5000);
                  }
                  else if (categories) {
                    if (!($(".modal-dialog").length > 0)) {

                        console.log("Categories: " + JSON.stringify(categories));
                        getCategory(categories, id);
                    }
                  }
                  else {
                   bootbox.alert("Oops! It seems the program can't extract content from this site.");
                        window.setTimeout(function() {
                          bootbox.hideAll();
                       }, 5000);
                   }
              } else {
                        bootbox.alert("Oops! It seems the program can't extract content from this site.");
                        window.setTimeout(function() {
                          bootbox.hideAll();
                       }, 5000);
                  }
            },
            error: function(e) {
              console.log(JSON.stringify(e));
              console.error("Error when sending ajax call");
              bootbox.alert("Oops! It seems this site doesn't have article content.");
                        window.setTimeout(function() {
                          bootbox.hideAll();
                       }, 5000);
            }
        })
        })
    })
  }
function getCategory(categories, id) {
    function counter($el, n) {
      (function loop() {
         $el.html(n);
         if (n--) {
             setTimeout(loop, 1000);
         }
      })();
    }

    var postCategory = '';
    var list_category = [];



    count = 0;
    for (var i = 0; i < categories.length; i++) {
      category_label = '<ul style="font-size:150%;text-align:center;margin-bottom:0px;">' + categories[i]["label"].toUpperCase() + '</ul>';
      category_score = '<ul class="list-group" style="font-size:100%;text-align:center">' + '(' + categories[i]["score"] + ')</ul>';
      postCategory  += (category_label + category_score);

    }
    // categories.forEach(function(category) {
    //   category = category["label"];
    //   if (typeof category == "string") {
    //       if (category == 'real-estate') {
    //         postCategory += '<span class="label label-primary" style="font-size:130%">' + 'Real Estate' + '</span>';
            
    //       } else if (category == "personal-finance") {
    //         postCategory += '<span class="label label-primary" style="font-size:130%">' + 'Personal Finance' + '</span>';
    //       }
    //       else {
    //           var labels = category.split('-');
    //           postCategory += '<span class="label label-primary" style="font-size:130%">' + labels[0][0].toUpperCase() + labels[0].slice(1);
    //           if (labels.length > 1) {
    //               for (var i = 1; i < labels.length; i++) {
    //                 postCategory += ' & ' + labels[i][0].toUpperCase() + labels[i].slice(1);
    //               }
    //           } else
    //             postCategory += "</span>";
    //       }
    //   }
    // })
    var message = '<div class="row">' +
                    '<div class="col-md-12"> ' +
                    '<form class="form-horizontal"> ' +
                    '<div class="form-group"> ' +

                    '<div> This article belongs to ' + postCategory + '</div>'


                    + '<div class="col-md-12"><hr></div>' +
                    '</div>' +

                    '<div class="form-group"> ' +
                    '<span>' + noti + '</span></div>' +

                    '<div class="form-group"> ' +
                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-0"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-0" value="art-entertainment"> Art & Entertaiment </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-1"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-1" value="automotive"> Automotive </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-2"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-2" value="business"> Business </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-3"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-3" value="career"> Career </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-4"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-4" value="education"> Education </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-5"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-5" value="family-parenting"> Family & Parenting </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-6"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-6" value="food-drink"> Food & Drink </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-7"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-7" value="health-fitness"> Health & Fitness </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-8"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-8" value="hobby-interest"> Hobbies & Interests </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-9"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-9" value="home-garden"> Home & Garden </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-10"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-10" value="law-gov-politics"> Law & Gov & Politics </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-11"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-11" value="personal-finance"> Personal Finance </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-12"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-12" value="pet"> Pet </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-13"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-13" value="religion-spirituality"> Religion & Spirituality </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-14"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-14" value="real-estate"> Real Estate </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-15"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-15" value="science"> Science </label></div></div>' +
                
                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-16"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-16" value="shopping"> Shopping </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-17"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-17" value="society"> Society </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-18"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-18" value="sport"> Sport </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-19"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-19" value="style-fashion"> Style & Fashion </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-20"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-20" value="technology"> Technology </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-21"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-21" value="travel"> Travel </label></div></div>' +

                    '<div class="col-md-4"> ' +
                    '<div class="checkbox"> <label for="awesomeness-22"> ' +
                    '<input type="checkbox" class="knorex-bookmarklet-input" class="knorex-bookmarklet" name="awesomeness[]" id="awesomeness-22" value="others"> Others </label></div></div></div>' +
                    '</form></div></div>' ;

    (function($) {
      var check_mess = $(message);

      categories.forEach(function(category) {
        category = category["label"];
        list_category.push(category);
        //Do not check messages
        //check_mess.find('input[value=' + category + ']').attr("checked", "checked");
    })
     
    message = check_mess.html();
    })(jQuery);
    

    
    bootbox.dialog({
                title: "Category classification result",
                message: message +  '</div><div class="form-group"><div class="col-md-3"></div><span class="alert alert-info" role="alert">This popup will disappear in &nbsp;' + '<span id="countdown" class="alert-link"></span>' +  '&nbsp; second</div></div>',
                show: true,
                backdrop: true,
                className: "dialog-position",
                buttons: {
                    success: {
                        label: "Send",
                        className: "btn-success",
                        callback: function () {
                            (function($) {

                                var answers = [];

                                $('input:checkbox.knorex-bookmarklet-input:checked').each(function(i){
                                  var selected_category = $(this).val();
                                  if (selected_category) {
                                      answers.push(selected_category);
                                  }
                                });
                              
                                var url = location.href;
                                data = {'url': url, 'categories_feedback': answers, 'categories': list_category, "_id": id};

                                 $.ajax({
                                      type: "GET",
                                      // url: 'http://54.255.101.212:19006/classifier/get_feedback',
                                      url: feedback_link + '/rest/send_feedback',
                                      dataType: 'jsonp',
                                      jsonp: 'callback',
                                      data: {'data': JSON.stringify(data)},
                                      success: function(data) {
                                        bootbox.hideAll();
                                      },
                                      error: function(e) {
                                        alert("error" + JSON.stringify(e));
                                        bootbox.hideAll();
                                      }
                                 })
                          })(jQuery);
                        }
                    }
                }
            }
    );

    jQuery(function($) {
        counter($('#countdown'), coundown_time);
        window.setTimeout(function() {
          bootbox.hideAll();
        }, coundown_time * 1000);
    })

}
