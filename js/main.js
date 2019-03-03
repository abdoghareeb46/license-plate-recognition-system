$(function(){
      $('.upload_img').on('submit',function (e) {
        e.preventDefault();
        get_res()
        // $(".succ").fadeOut(100);
        // $(".fail").fadeIn(500);
      })

      $("#imgInp").change(function() {
        readURL(this);
      });
})

function readURL(input) {

  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function(e) {
      $('#myImage').attr('src', e.target.result);
    }

    reader.readAsDataURL(input.files[0]);
  }
}

function get_res(){
  var form_data = new FormData($('.upload_img')[0]);
        $.ajax({
            type: 'post',
            url: 'http://127.0.0.1:9090/Process',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                console.log(data);
                $.each(data["imges"], function (index, val) { 
                  img="<img src='"+val+"' class='col-md-3' style='margin:10px'>"
                  $(".segmentedimages").append(img);
                });
               if(data['res']==true){
                $(".succ").fadeIn(500);
                $(".fail").fadeOut(5);
              }else if(data['res']==false){
                $(".succ").fadeOut(5);
                $(".fail").fadeIn(500);
              }else{
                $(".succ").fadeOut(5);
                $(".fail").fadeOut(5);
              }
            },
        });
}
