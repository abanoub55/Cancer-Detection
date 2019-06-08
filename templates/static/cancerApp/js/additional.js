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
                $('#lung_img').attr('src', "../static/cancerApp/img/login.jpg");
                $('#lung_img').hide();
                $('#lung_img').fadeIn(650);
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
            timeout:0,
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            url: 'prediction',
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
           error:function(){
           $('.loader').hide();
           alert("Error! something went wrong :(");
           window.location.reload();
           },
        });
    });

    var radioID;
     // Visualization
     $('input[type="radio"]').click(function(event){
            radioID = event.target.id;
            console.log(radioID);
     });

    $('#btn-visualize').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        form_data.append('image',$('#upload-file')[0]);
        if($('#'+radioID).prop("checked") == true){
        $('#btn-visualize').hide();
        $('#lung_img').attr('src', "../static/cancerApp/img/login.jpg");
        $('.loader').show();
        $.ajax({
            type: 'POST',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            url: radioID,
            success: function (data) {
                // Get and display2 the result
                $('.loader').hide();
                $('#btn-visualize').show();
                if(data!=='healthy'){
                d = new Date();
                $('#lung_img').attr('src', "../static/cancerApp/img/lungfig.jpg?"+d.getTime());
                    }
                  else{
                        alert('patient is healthy')
                    }

            },
            error:function(){
            $('.loader').hide();
            alert("Error! something went wrong :(");
            window.location.reload();
            },
        });
        }else{
            alert('Error! No option specified')
        }

    });

    //Make Statistics
    $('input[type="checkbox"]').click(function(event){
            var id = event.target.id;
            img_dest = id+'_chart.jpg';
            div_id = id+"-div";
            img_id = id+"-img";
            if($(this).prop("checked") == true){
            $.ajax({
            type: 'GET',
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            url: id+'Stats',
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

