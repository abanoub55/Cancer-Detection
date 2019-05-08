$(document).ready(
function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    createStats();

    function createStats(){
    let list =  $('input[type=checkbox]');
    let i =0;
    while (i<list.length){
        let div = document.createElement('div');
        div.className="statDiv";
        let main_div = document.querySelector('.main-div');
        let img = document.createElement('img');
        div.appendChild(img);
        img.src="#";
        img.alt = "loading "+list[i].id+" chart"
        img.id = list[i].id+'-img';
        div.id = list[i].id+"-div";
        console.log(div);
        main_div.appendChild(div);
        $('#'+div.id).hide();
        i++;
        }
    }

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
        $('#imageUpload').hide();
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


     // Visualization
    $('#btn-visualize').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        form_data.append('image',$('#upload-file')[0])
        // Show loading animation


        // Make visualization by calling api /visualize
        if(document.getElementById('visualizeLung').checked){
        $('#btn-visualize').hide();
    	$('.loader').show();
        $.ajax({
            type: 'POST',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            url: 'lungStructure',
            success: function (data) {
                // Get and display2 the result
                $('.loader').hide();
                $('#btn-visualize').show();
                $('#result').fadeIn(600);
                $('#imagePreview2').css('background-image', "url('http://127.0.0.1:8000/static/cancerApp/img/lungfig.jpg')");
                console.log('Lung Structure!');
            },
        });
  //  });
    }
    else if(document.getElementById('visualizeRib').checked){
      	$('#btn-visualize').hide();
        $('.loader').show();
        $.ajax({
            type: 'POST',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            url: 'ribVisualize',
            success: function (data) {
                // Get and display2 the result
                $('.loader').hide();
                $('#btn-visualize').show();
                $('#result').fadeIn(600);
                $('#imagePreview2').css('background-image', "url('http://127.0.0.1:8000/static/cancerApp/img/lungfig.jpg')");
                console.log("Rib Visulization");
            },
        });
    }
    else {
    	//$('.loader').hide();
    	window.alert("No option was specified!")

    };
    });

    //Make Statistics
    $('input[type="checkbox"]').click(function(event){
            var id = event.target.id;
            img_dest = id+'_chart.jpg';
            div_id = id+"-div";
            img_id = id+"-img";
            var idUrl;
            if(id === 'cancer'){
                idUrl='cancerStats';
            }
            else if(id === 'gender'){
                idUrl = 'genderStats';
            }
            else if(id === 'age'){
                idUrl = 'ageStats';
            }
            if($(this).prop("checked") == true){
            $.ajax({
            type: 'GET',
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            url: idUrl,
            success: function (data) {
            if(data == "user has no activity yet"){$('#'+div_id).hide();$('#'+id).prop("checked",false);window.alert(data);}
           else{
            $('#'+div_id).show();
            $('#'+img_id).attr('src','../static/cancerApp/img/'+img_dest);
            }
            console.log('Success!');
            },
                });
            }
            else{
                $('#'+div_id).hide();
            }

        });

    $('#h_clear').click(function(){
     $.ajax({
            type: 'GET',
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            url: 'clearHistory',
            success: function (data) {
            alert(data);
            },
        });
    });

});

