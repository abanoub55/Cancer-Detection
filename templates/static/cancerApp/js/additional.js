$(document).ready(
function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', "url('http://127.0.0.1:8000/static/cancerApp/img/prediction.jpg')");
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
                $('#imagePreview2').css('background-image', "url('http://127.0.0.1:8000/static/cancerApp/img/login.jpg')");
                $('#imagePreview2').hide();
                $('#imagePreview2').fadeIn(650);

            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#btn-visualize').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        form_data.append('image',$('#upload-file')[0])
        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            url: '/predict',
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data.toString());
                if(data.toString()=='patient is suspected to have cancer')
                {
                    $('#imagePreview').css("background-image", "url('http://127.0.0.1:8000/static/cancerApp/img/unhealthy.jpg')");
                }
                else
                {
                    $('#imagePreview').css('background-image', "url('http://127.0.0.1:8000/static/cancerApp/img/healthy.jpg')");

                }
                console.log('Success!');
            },
        });
    });

    $('#btn-visualize').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        form_data.append('image',$('#upload-file')[0])
        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            url: 'visualizeFn',
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#imagePreview2').css('background-image', "url('http://127.0.0.1:8000/static/cancerApp/img/lungfig.jpg')");
                console.log('Success!');
            },
        });
    });

      $('#btn-stats').click(function () {
        $.ajax({
            type: 'GET',
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            url: 'showStats',
            success: function (data) {
            if(data == "user has no activity yet"){ window.alert(data);}
           else{ document.getElementById('img').src = "../static/cancerApp/img/chart.jpg";}
            console.log('Success!');
            },
        });
    });

});

